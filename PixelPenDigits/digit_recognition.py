import sys
import pygame
import numpy as np
import tensorflow as tf

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 280
SCREEN_HEIGHT = 280
GRID_SIZE = 28
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Load the pre-trained model
model = tf.keras.models.load_model(sys.argv[1])

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Digit Recognition")

# Font for displaying prediction
font = pygame.font.SysFont(None, 60)

# Function to draw on the screen
def draw_grid():
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

def draw_pixel(row, col):
    pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE * 2, CELL_SIZE * 2))

def main():
    # Main loop
    running = True
    drawing = False
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    row, col = event.pos[1] // CELL_SIZE, event.pos[0] // CELL_SIZE
                    grid[row][col] = 1
                    draw_pixel(row, col)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset grid
                    grid.fill(0)
                    screen.fill(WHITE)
                    draw_grid()
                elif event.key == pygame.K_c:  # Check prediction
                    # Reshape for model input
                    digit_data = np.reshape(grid, (1, GRID_SIZE, GRID_SIZE, 1))

                    # Predict the digit
                    prediction = model.predict(digit_data)
                    digit = np.argmax(prediction)

                    # Print predicted digit
                    print("Predicted digit:", digit)

                    # Display prediction
                    text = font.render(str(digit), True, RED)
                    screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
                    pygame.display.flip()

        # Update the display
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python digit_recognition.py <model_file>")
    else:
        screen.fill(WHITE)
        draw_grid()
        pygame.display.update()
        main()


