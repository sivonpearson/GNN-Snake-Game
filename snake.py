# importing libraries
import os
import pygame
import time
import random

INITIAL_LENGTH = 4
TILE_SIZE = 25
BOARD_SIZE = 15
INITIAL_BODY_LEN = 3
WINDOW_X = 900
WINDOW_Y = 200

# Converts binary number to index location
def bin2idx(b):
    return b.bit_length() - 1

# Converts index location to binary number
def idx2bin(index):
    return 1 << index

class Board:
    def __init__(self):
        self.side_len = BOARD_SIZE
        self.area = self.side_len ** 2

        self.reset()

    def reset(self):
        self.walls = 0

        for i in range(self.side_len):
            self.walls += idx2bin(i)
            self.walls += idx2bin(self.side_len * (self.side_len - 1) + i)

        for i in range(self.side_len-2):
            self.walls += idx2bin(self.side_len + self.side_len * i)
            self.walls += idx2bin((self.side_len << 1) - 1 + self.side_len * i)

        self.food = 0
        self.head = idx2bin(round(self.area >> 1)) << self.side_len
        self.body_list = [(self.head << i) for i in range(INITIAL_BODY_LEN, 0, -1)]
        self.facing = 0 # rightward

        self.food_eaten = 0
        self.num_moves = 0

        self.end = False

        self.update()

        self.place_food()
        
    def update(self):
        self.body = sum(self.body_list)

        self.all = self.walls | self.food | self.body | self.head

        if self.head & (self.body | self.walls):
            self.end = True

    def place_food(self):
        choices = []
        for i in range(self.area):
            if idx2bin(i) & self.all == 0:
                choices.append(i)

        random_location = random.choice(choices)
        self.food = idx2bin(random_location)

    def get_binstring(self,var):
        con = '{:'+str(self.area)+'b}'
        return con.format(var)

    # Moves snake
    def push(self, move):
        old_head = self.head

        # 0 - right (00)
        # 1 - up    (01)
        # 2 - left  (10)
        # 3 - down  (11)

        # must not be able to change direction if it is opposite

        # it is assumed that: 0 <= move <= 3
        if move is not None and (move & 1) ^ (self.facing & 1):
            self.facing = move

        match self.facing:
            case 0: # right
                self.head >>= 1
            case 1: # up
                self.head <<= self.side_len
            case 2: # left
                self.head <<= 1
            case 3: # down
                self.head >>= self.side_len

        # if landing on empty tile
        if self.head & self.food == 0:
            self.body_list.remove(self.body_list[0])
        # if landing on tile with food
        else:
            self.food_eaten += 1
            self.place_food()
        self.body_list.append(old_head)

        self.update()
        self.num_moves += 1

import pygame

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
LIMEGREEN = pygame.Color(0, 255, 0)
DARKGREEN = pygame.Color(0, 150, 0)
BLUE = pygame.Color(0, 0, 255)

class Display:
    """
    Parameters
    ----------
    frame_rate: Update speed for the game.

    userfocus: If True, it will give the user a second to prepare as well as show the gameover screen when the user loses.

    autoreset: If True, it will reset the game automatically.
    """
    def __init__(self, frame_rate = 10, userfocus = False, autoreset = False):
        self.state = Board()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" % (WINDOW_X, WINDOW_Y)
        pygame.init()
        pygame.display.set_caption('Snake game')
        self.window_size = BOARD_SIZE * TILE_SIZE
        self.frame_rate = frame_rate
        self.game_window = pygame.display.set_mode((self.window_size, self.window_size))
        self.fps = pygame.time.Clock()
        self.userfocus = userfocus
        self.autoreset = autoreset

    def stop(self):
        pygame.quit()

    def game_over_text(self):
        game_end_font = pygame.font.SysFont('times new roman', 40)
        game_end_surface = game_end_font.render('Final score: ' + str(self.state.food_eaten), True, RED)
        game_end_rect = game_end_surface.get_rect()
        game_end_rect.midtop = (self.window_size/2, self.window_size/4)
        self.game_window.blit(game_end_surface, game_end_rect)
        pygame.display.flip()
        time.sleep(0.75)

    def show(self, input_func, stop_loops = False, step_food_ratio = 100):
        score_font = pygame.font.SysFont('times new roman', 25)

        while True:
            # helps user get ready
            if self.userfocus:
                time.sleep(1)

            while not self.state.end:
                pygame.event.pump()

                self.game_window.fill(BLACK)
                # convert the binary Snake board to a version that is displayable
                walls = self.state.get_binstring(self.state.walls)
                head = self.state.get_binstring(self.state.head)
                food = self.state.get_binstring(self.state.food)
                body = self.state.get_binstring(self.state.body)

                for i in range(self.state.area):
                    x_pos = i % BOARD_SIZE
                    y_pos = i // BOARD_SIZE

                    if walls[i] == '1':
                        pygame.draw.rect(self.game_window, RED, pygame.Rect(x_pos * TILE_SIZE, y_pos * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    elif head[i] == '1':
                        pygame.draw.rect(self.game_window, LIMEGREEN, pygame.Rect(x_pos * TILE_SIZE, y_pos * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    elif food[i] == '1':
                        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(x_pos * TILE_SIZE, y_pos * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    elif body[i] == '1':
                        pygame.draw.rect(self.game_window, DARKGREEN, pygame.Rect(x_pos * TILE_SIZE, y_pos * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        
                score_surface = score_font.render('Score : ' + str(self.state.food_eaten), True, WHITE)
                score_rect = score_surface.get_rect()
                self.game_window.blit(score_surface, score_rect)

                move = input_func(self.state)

                self.state.push(move)
                pygame.display.update()
                self.fps.tick(self.frame_rate)

                if stop_loops and (self.state.num_moves + (1. + self.state.food_eaten)) > step_food_ratio:
                    break

            if self.userfocus:
                self.game_over_text()

            if self.autoreset:
                self.state.reset()
            else:
                break

def user_input_func(param=None):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            match event.key:
                case pygame.K_UP:
                    return 1
                case pygame.K_DOWN:
                    return 3
                case pygame.K_RIGHT:
                    return 0
                case pygame.K_LEFT:
                    return 2
    return None

if __name__ == '__main__':
    screen = Display(frame_rate=8, userfocus=True, autoreset=True)
    screen.show(input_func=user_input_func)