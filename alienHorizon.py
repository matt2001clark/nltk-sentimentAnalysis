# Simple pygame program

# Import and initialize the pygame library
import pygame
import random
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([1000, 500])

r = random.randint(0,125)
g = random.randint(0,125)
b = random.randint(0,125)
rgb = [r,g,b]

r2 = random.randint(0,255)
g2 = random.randint(0,255)
b2 = random.randint(0,255)
rgb2 = [r2,g2,b2]

r3 = random.randint(125,255)
g3 = random.randint(125,255)
b3 = random.randint(125,255)
rgb3 = [r3,g3,b3]

cx = random.randint(50,950)
cy = random.randint(0,400)
size1 = random.randint(50,200)

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a circle and rectangle(hor x250,vert y250,)
    
    pygame.draw.rect(screen, (rgb3), pygame.Rect(0,0,1000,500))
    pygame.draw.circle(screen, (rgb), (cx, cy), size1)
    pygame.draw.rect(screen, (rgb2), pygame.Rect(0,300,1000,200))

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
