import pygame
import random

### initialisation
pygame.init()
window = pygame.display.set_mode( ( 1000, 500 ) )
pygame.display.set_caption("Gradient Rect")

def gradientRect( window, left_colour, right_colour, target_rect ):
    """ Draw a horizontal-gradient filled rectangle covering <target_rect> """
    colour_rect = pygame.Surface( ( 2, 2 ) )                                   # tiny! 2x2 bitmap
    pygame.draw.line( colour_rect, left_colour,  ( 0,0 ), ( 0,1 ) )            # left colour line
    pygame.draw.line( colour_rect, right_colour, ( 1,0 ), ( 1,1 ) )            # right colour line
    colour_rect = pygame.transform.smoothscale( colour_rect, ( target_rect.width, target_rect.height ) )  # stretch!
    window.blit( colour_rect, target_rect )                                    # paint it

r = random.randint(0,125)
g = random.randint(0,125)
b = random.randint(0,125)
rgb = [r,g,b]

r2 = random.randint(0,125)
g2 = random.randint(0,125)
b2 = random.randint(0,125)
rgb2 = [r2,g2,b2]

r3 = random.randint(125,255)
g3 = random.randint(125,255)
b3 = random.randint(125,255)
rgb3 = [r3,g3,b3]

r4 = random.randint(125,255)
g4 = random.randint(125,255)
b4 = random.randint(125,255)
rgb4 = [r4,g4,b4]

r5 = random.randint(0,255)
g5 = random.randint(0,255)
b5 = random.randint(0,255)
rgb5 = [r5,g5,b5]


cx = random.randint(50,950)
cy = random.randint(0,400)
size1 = random.randint(50,200)

### Main Loop
clock = pygame.time.Clock()
finished = False
while not finished:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finished = True

    # Handle user-input
    for event in pygame.event.get():
        if ( event.type == pygame.QUIT ):
            finished = True

    # Update the window
    window.fill( ( 0,0,0 ) )
    gradientRect( window, rgb, rgb2, pygame.Rect( 0,0, 1000, 500 ) )
    pygame.draw.circle(window, (rgb5), (cx, cy), size1)
    gradientRect( window, rgb3, rgb4, pygame.Rect( 0,300, 1000, 200 ) )
    pygame.display.flip()

    # Clamp FPS
    clock.tick_busy_loop(60)

pygame.quit()
