"""
Flappy Bird Game with PID Controller

This module implements a Flappy Bird-like game where the player controls a bird
that must navigate through pipes. The game features both manual control via keyboard
and an optional AI control using a PID controller.
"""

import copy
import random
import pygame
from dataclasses import dataclass
from typing import Tuple, Optional


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

    def get_rects(
        self, transform_func, screen_height: int
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        """Get the pipe's rectangles for rendering and collision detection.

        Args:
            transform_func: Function to transform game coordinates to screen coordinates
            screen_height: Height of the screen

        Returns:
            Tuple of (bottom_pipe_rect, top_pipe_rect)
        """
        x_screen, y_screen = transform_func(self.x, self.h)
        bottom_pipe_rect = pygame.Rect(x_screen, y_screen, self.w, self.h)
        top_pipe_rect = pygame.Rect(
            x_screen, 0, self.w, screen_height - self.h - self.gap
        )
        return bottom_pipe_rect, top_pipe_rect


def pipe_motion(
    pipe: Pipe, vx: float, dt: float, screen_width: int = SCREEN_WIDTH
) -> Tuple[Pipe, int]:
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


def check_collision(
    bird_rect: pygame.Rect,
    bottom_pipe_rect: pygame.Rect,
    top_pipe_rect: pygame.Rect,
    bird_y: float,
    bird_h: float,
) -> bool:
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
        bird_rect.colliderect(bottom_pipe_rect)
        or bird_rect.colliderect(top_pipe_rect)
        or bird_y + bird_h > 1.5 * SCREEN_HEIGHT
        or bird_y < -0.5 * SCREEN_HEIGHT
    )


def draw_game(
    screen: pygame.Surface,
    bird_rect: pygame.Rect,
    bottom_pipe_rect: pygame.Rect,
    top_pipe_rect: pygame.Rect,
    score: int,
    user_mode: bool,
) -> None:
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

    mode_switch_text = small_font.render(
        "Press M to toggle Auto/User mode", True, BLACK
    )
    screen.blit(mode_switch_text, (10, SCREEN_HEIGHT - 25))



@dataclass
class PIDController:
    Kp: float = 1.5  # Proportional gain
    Ki: float = 0.01  # Integral gain
    Kd: float = 0.5  # Derivative gain
    error_accumulator: float = 0  # Accumulated error for integral term
    prev_error: float = 0  # Previous error for derivative term
    max_accumulator: float = 200  # Anti-windup limit
    dt: float = 1 / FPS  # Time step for calculations

    def reset(self) -> None:
        """Reset the controller state."""
        self.error_accumulator = 0
        self.prev_error = 0

    def pid_optimizer(self, set_point, process_var,umin, umax):
        e = set_point - process_var
        P = self.Kp * e

        self.error_accumulator += e
        I = self.Ki * self.error_accumulator

        D = self.Kd * (e - self.prev_error)
        self.prev_error = e

        pid = P + I + D
        if pid < umin:
            return umin
        elif pid > umax:
            return umax
        else:
            return pid
   
    def calc_input(
        self,
        set_point: float,
        process_var: float,
        velocity: float = 0,
        umin: float = -500,
        umax: float = 500,
    ) -> float:
        """Calculate the control signal with improved anti-windup and velocity feedback.

        Args:
            set_point: Target value
            process_var: Current measured value
            velocity: Current velocity (for feedforward control)
            umin: Minimum control output
            umax: Maximum control output

        Returns:
            Control signal value
        """

        return self.pid_optimizer(set_point, process_var, umin, umax)


def calculate_control_signal(bird: Bird, pipe: Pipe, pid: PIDController) -> float:
    """Calculate the control signal for the bird using PID controller.

    Args:
        bird: Current bird state
        pipe: Current pipe state
        pid: PID controller instance

    Returns:
        Control signal value
    """
    # Only consider pipes that are ahead of the bird
    if pipe.x + pipe.w < bird.x:
        return 0

    # Calculate target height (i.e., the middle of the gap)
    target_height = pipe.h + (pipe.gap / 2.0) + bird.h

    # Adjust target height based on bird's velocity
    velocity_offset = bird.vy * 0.5

    adjusted_target = target_height - velocity_offset

    # Get current height and distance to pipe
    current_height = bird.y + bird.h / 2
    distance_to_pipe = pipe.x - (bird.x + bird.w)

    # Adjust control based on distance to pipe
    # More aggressive control when closer to pipe
    # Ensure we don't divide by zero
    if distance_to_pipe <= -SCREEN_WIDTH/2:
        distance_factor = 1.5  # Maximum control factor when very close or past pipe
    else:
        distance_factor = max(0.5, min(1.5, 1 + 1 / (distance_to_pipe + SCREEN_WIDTH/2 + 1e-8)))

    # Calculate and return control signal with bird's velocity for better derivative control
    return pid.calc_input(adjusted_target, current_height, bird.vy) * distance_factor



def main():
    """Main game function."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird Game")

    def transform(x, y):
        """Helper function to convert game coordinates to screen coordinates"""
        return (x, SCREEN_HEIGHT - y)

    # Initialize the PID controller with optimized parameters
    pid = PIDController(2.5, 0.001, 1.5)

    # Initialize the game states
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

    # Main game loop
    while running:
        # Process all events
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
                    pid.reset()
                    print(f"Switched to {'user' if user_mode else 'AI'} control mode")

        # Check for held keys for more responsive control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            jump_force = JUMP_FORCE
            user_input = True

        # Determine control signal based on mode
        if user_mode:
            u_jump = jump_force
        else:
            u_jump = calculate_control_signal(bird, pipe, pid)

        # Update the game states
        bird = bird_motion(bird, u_jump, dt)
        bird_rect = bird.get_rect(transform)

        pipe, d_score = pipe_motion(pipe, bird.vx, dt)
        score += d_score

        if d_score > 0:
            bird.vx += PIPE_SPEED_INCREASE

        # Draw game elements
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
