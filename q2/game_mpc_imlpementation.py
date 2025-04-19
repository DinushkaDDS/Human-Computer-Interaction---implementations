"""
Flappy Bird Game with Model Predictive Controller

This module implements a Flappy Bird-like game where the player controls a bird
that must navigate through pipes. The game features both manual control via keyboard
and an optional AI control using a Model Predictive Controller (MPC).
"""

import copy
import random
import pygame
import numpy as np  # Used for candidate control generation
from dataclasses import dataclass
from typing import Tuple, Optional
import pyomo.environ as pyo

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = -80
JUMP_FORCE = 800
PIPE_SPEED_INCREASE = 5

@dataclass
class Bird:
    x: float  # x position
    y: float  # y position
    vx: float  # x velocity
    vy: float  # y velocity
    w: float = 20  # width
    h: float = 20  # height

    def get_rect(self, transform_func) -> pygame.Rect:
        """Get the bird's rectangle for rendering and collision detection."""
        x, y = transform_func(self.x, self.y)
        return pygame.Rect(x, y, self.w, self.h)

def bird_motion(bird: Bird, u: float, dt: float, gravity: float = GRAVITY) -> Bird:
    """Updates the bird's position and velocity.

    Args:
        bird: Bird object to update
        u: Control input (upward force)
        dt: Time step
        gravity: Gravitational constant

    Returns:
        Updated Bird object
    """
    new_bird = copy.deepcopy(bird)
    if u > 0:
        # Reset downward velocity when jumping for more responsive feel
        new_bird.vy = 0

    # Update position and velocity
    new_bird.y = bird.y + bird.vy * dt
    new_bird.vy = bird.vy + (u + gravity) * dt

    return new_bird

@dataclass
class Pipe:
    """Represents an obstacle pipe pair."""
    x: float
    h: float
    w: float = 70
    gap: float = 200

    def get_rects(self, transform_func, screen_height: int) -> Tuple[pygame.Rect, pygame.Rect]:
        """Get the pipe's rectangles for rendering and collision detection.

        Args:
            transform_func: Function to transform game coordinates to screen coordinates
            screen_height: Height of the screen

        Returns:
            Tuple of (bottom_pipe_rect, top_pipe_rect)
        """
        x_screen, y_screen = transform_func(self.x, self.h)
        bottom_pipe_rect = pygame.Rect(x_screen, y_screen, self.w, self.h)
        top_pipe_rect = pygame.Rect(x_screen, 0, self.w, screen_height - self.h - self.gap)
        return bottom_pipe_rect, top_pipe_rect

def pipe_motion(pipe: Pipe, vx: float, dt: float, screen_width: int = SCREEN_WIDTH) -> Tuple[Pipe, int]:
    """Updates the pipe position and generates new pipes when needed.

    Args:
        pipe: Pipe object to update
        vx: Horizontal velocity
        dt: Time step
        screen_width: Width of the screen

    Returns:
        Tuple of (updated Pipe object, score increment)
    """
    new_pipe = copy.deepcopy(pipe)
    new_pipe.x -= vx * dt

    d_score = 0
    if new_pipe.x < -pipe.w:
        new_pipe.x = screen_width
        new_pipe.h = random.randint(200, 300)
        d_score = 1
    return new_pipe, d_score

def check_collision(bird_rect: pygame.Rect,
                    bottom_pipe_rect: pygame.Rect,
                    top_pipe_rect: pygame.Rect,
                    bird_y: float,
                    bird_h: float) -> bool:
    """Check if the bird has collided with pipes or gone out of bounds.

    Args:
        bird_rect: Bird's rectangle
        bottom_pipe_rect: Bottom pipe's rectangle
        top_pipe_rect: Top pipe's rectangle
        bird_y: Bird's y position
        bird_h: Bird's height

    Returns:
        True if collision detected, False otherwise
    """
    return (
        bird_rect.colliderect(bottom_pipe_rect) or
        bird_rect.colliderect(top_pipe_rect) or
        bird_y + bird_h > 1.5 * SCREEN_HEIGHT or
        bird_y < -0.5 * SCREEN_HEIGHT
    )

def draw_game(screen: pygame.Surface,
              bird_rect: pygame.Rect,
              bottom_pipe_rect: pygame.Rect,
              top_pipe_rect: pygame.Rect,
              score: int,
              user_mode: bool) -> None:
    """Draw all game elements.

    Args:
        screen: Pygame surface to draw on
        bird_rect: Bird's rectangle
        bottom_pipe_rect: Bottom pipe's rectangle
        top_pipe_rect: Top pipe's rectangle
        score: Current score
        user_mode: Whether the game is in user control mode
    """
    WHITE = (240, 240, 240)
    GREEN = (0, 200, 0)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 200)
    RED = (200, 0, 0)

    screen.fill(WHITE)
    pygame.draw.rect(screen, GREEN, bird_rect)
    pygame.draw.rect(screen, GREEN, bottom_pipe_rect)
    pygame.draw.rect(screen, GREEN, top_pipe_rect)

    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(text, (10, 10))

    mode_text = font.render(f"Mode: {'USER' if user_mode else 'AUTO'}", True, BLACK)
    screen.blit(mode_text, (SCREEN_WIDTH - 150, 10))

    small_font = pygame.font.Font(None, 24)
    controls_text = small_font.render("Press SPACE to jump", True, BLACK)
    screen.blit(controls_text, (10, SCREEN_HEIGHT - 50))

    mode_switch_text = small_font.render("Press M to toggle Auto/User mode", True, BLACK)
    screen.blit(mode_switch_text, (10, SCREEN_HEIGHT - 25))

@dataclass
class MPCController:
    horizon: int = 10      # Prediction horizon (number of time steps)
    dt: float = 1 / FPS    # Time step for simulation
    umin: float = 0        # Minimum control input (no force)
    umax: float = JUMP_FORCE  # Maximum control input (jump force)
    num_candidates: int = 11  # Number of candidate control actions

    def calc_input(self, bird: Bird, pipe: Pipe) -> float:
        """Calculate the control signal using a Pyomo-based MPC formulation.
        
        The model predicts the bird's vertical states over a horizon using its dynamics:
        
            y[t+1] = y[t] + v[t]*dt  
            v[t+1] = v[t] + (u[t] + GRAVITY)*dt  
        
        The objective is to minimize the sum of squared differences between the predicted
        bird center (y[t] + bird.h/2) and the target height (middle of the pipe gap) over the horizon,
        along with a small penalty on control effort.

        Here no need to have a reference trajectory because for a any given point the best trajectory is straight up.
        
        Args:
            bird: Current state of the bird.
            pipe: Current pipe state.
            
        Returns:
            The first control input from the optimal control sequence.
        """

        # Set target height: middle of the pipe gap
        target_height = pipe.h + pipe.gap / 2.0

        horizon = self.horizon
        dt = self.dt  # use the controller's time step

        model = pyo.ConcreteModel()

        # Define index sets: time steps 0...horizon for states, 0...horizon-1 for controls.
        model.T = pyo.RangeSet(0, horizon)

        # Decision variables: y and v for states, u for control inputs.
        model.y = pyo.Var(model.T)
        model.vy = pyo.Var(model.T)
        
        # Control input bounds: assume u ∈ [0, JUMP_FORCE + GRAVITY] (only upward thrust is applied)
        model.ctrl = pyo.Var(model.T, domain=pyo.Binary)
        model.u = pyo.Var(model.T, bounds=(0, JUMP_FORCE + GRAVITY))

        # Initial conditions: bird's current y and velocity.
        model.y[0].fix(bird.y)
        model.vy[0].fix(bird.vy)

        # Condition to make sure control input is either active or deactive
        def int_rule(m, t):
            return m.u[t] == JUMP_FORCE*m.ctrl[t]
        model.binary_rule = pyo.Constraint(model.T, rule=int_rule)

        # Dynamics constraints: update state based on discrete time model.
        def y_dy_rule(m, t):
            if t < horizon:
                return m.y[t + 1] == m.y[t] + m.vy[t] * dt
            return pyo.Constraint.Skip
        model.y_dy = pyo.Constraint(model.T, rule=y_dy_rule)

        def vy_dy_rule(m, t):
            if t < horizon:
                return m.vy[t + 1] == m.vy[t] + (m.u[t] + GRAVITY) * dt
            return pyo.Constraint.Skip
        model.vy_dy = pyo.Constraint(model.T, rule=vy_dy_rule)

        # Objective: minimize squared tracking error (bird center vs. target) plus a small control penalty(required).
        alpha = 0.0001
        beta = 0.1
        model.obj = pyo.Objective(
            expr=sum((model.y[t] + bird.h / 2 - target_height) ** 2 for t in model.T) 
                                        + alpha * sum(model.u[t] ** 2 for t in model.T) + beta * sum(model.vy[t]**2 for t in model.T) ,
            sense=pyo.minimize,
        )

        # Solve the model using an appropriate solver (e.g., IPOPT for nonlinear problems).
        solver = pyo.SolverFactory("ipopt")
        solution = solver.solve(model, tee=False)

        # Retrieve and return the first control input.
        u0 = [JUMP_FORCE*pyo.value(model.ctrl[t]) for t in model.T]
        return u0

def calculate_control_signal(bird: Bird, pipe: Pipe, mpc: MPCController) -> float:
    """Calculate the control signal for the bird using the MPC controller.

    Args:
        bird: Current bird state.
        pipe: Current pipe state.
        mpc: MPC controller instance.

    Returns:
        Control signal value.
    """
    # Only consider pipes that are ahead of the bird.
    if pipe.x + pipe.w < bird.x:
        return [0]

    # Use the MPC controller to compute the optimal control input.
    return mpc.calc_input(bird, pipe)


def main():
    """Main game function."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird Game with MPC")

    def transform(x, y):
        """Helper function to convert game coordinates to screen coordinates"""
        return (x, SCREEN_HEIGHT - y)
    
    PREDICTIVE_HORIZON = 10
    # Initialize the MPC controller with parameters.
    mpc = MPCController(horizon=PREDICTIVE_HORIZON, dt=1 / FPS, umin=0, umax=JUMP_FORCE)

    # Initialize the game states.
    user_mode = False
    bird = Bird(50, 300, 30, 0)
    bird_rect = bird.get_rect(transform)
    pipe_height = random.randint(200, 300)
    pipe = Pipe(SCREEN_WIDTH - 50, pipe_height)
    bottom_pipe_rect, top_pipe_rect = pipe.get_rects(transform, SCREEN_HEIGHT)

    clock = pygame.time.Clock()
    running = True
    dt = 1 / FPS
    score = 0

    u_jump_seq = calculate_control_signal(bird, pipe, mpc)

    # Main game loop.
    while running:
        events = pygame.event.get()
        running = True
        user_input = False
        jump_force = 0

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    jump_force = JUMP_FORCE
                    user_input = True
                elif event.key == pygame.K_m:
                    user_mode = not user_mode
                    print(f"Switched to {'user' if user_mode else 'AI'} control mode")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            jump_force = JUMP_FORCE
            user_input = True

        # Determine control signal based on mode.
        if user_mode:
            u_jump = jump_force
        else:
            # Only do the calculation at PREDICTIVE_HORIZON
            if(len(u_jump_seq)==0):
                u_jump_seq = calculate_control_signal(bird, pipe, mpc)
            u_jump = u_jump_seq.pop(0)

        # Update game states.
        bird = bird_motion(bird, u_jump, dt)
        bird_rect = bird.get_rect(transform)
        pipe, d_score = pipe_motion(pipe, bird.vx, dt)
        score += d_score

        if d_score > 0:
            bird.vx += PIPE_SPEED_INCREASE

        bottom_pipe_rect, top_pipe_rect = pipe.get_rects(transform, SCREEN_HEIGHT)
        draw_game(screen, bird_rect, bottom_pipe_rect, top_pipe_rect, score, user_mode)

        if check_collision(bird_rect, bottom_pipe_rect, top_pipe_rect, bird.y, bird.h):
            running = False

        pygame.display.update()
        clock.tick(FPS)

    pygame.time.delay(1000)
    pygame.quit()

if __name__ == "__main__":
    main()
