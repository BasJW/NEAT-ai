import os
import pygame
import random
import neat

screen = pygame.display.set_mode((600, 600))
random_colour = (100, 200, 100)
ground = (0, 0, 255)
pygame.font.init()
myfont = pygame.font.SysFont('Sans', 30)

global gen
gen = 0


class Player:

    def __init__(self):
        self.isJump = False
        self.JumpCount = 10
        self.x = 20
        self.y = 480
        self.acc = 1
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        self.colour = (r, g, b)

    def draw(self):
        pygame.draw.rect(screen, self.colour, (self.x, self.y, 60, 70))

    def jump(self):

        if self.isJump:
            if self.JumpCount >= -10:
                neg = 1
                if self.JumpCount < 0:
                    neg = -1
                self.y -= self.JumpCount ** 2 * 0.4 * neg
                self.JumpCount -= 1
            else:
                self.isJump = False
                self.JumpCount = 10


class Obstacle:
    def __init__(self):
        self.move_amount = random.randrange(15, 25)
        self.x = 550
        self.score = 0

    def draw(self):
        pygame.draw.rect(screen, (255, 0, 0), (self.x, 510, 50, 40))
        self.x -= self.move_amount

    def move(self):
        if self.x < -40:
            self.score += 1
            self.x = 550
            self.move_amount = random.randrange(13, 23)

    def collision(self, p_object):
        if self.x in range(20, 90) or self.x + 50 in range(20, 90):  # done
            if p_object.y + 70 in range(510, 551):
                return True  # returns true if collided

    def show_score(self):
        print(self.score)


def eval_genomes(genomes, config):
    nets = []
    ge = []
    players = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        players.append(Player())
        g.fitness = 0
        ge.append(g)

    block = Obstacle()
    global gen
    gen += 1
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                quit()

        clock = pygame.time.Clock()
        clock.tick(40)
        screen.fill((255, 255, 255))

        if len(players) <= 0:
            break

        for x, player in enumerate(players):
            player.draw()
            player.jump()
            ge[x].fitness += 0.1

            output = nets[x].activate((player.x+35, block.x, block.move_amount))
            if output[0] > 0.5:
                player.isJump = True

        block.draw()


        pygame.draw.rect(screen, ground, (0, 550, 600, 50))

        textsurface = myfont.render("Score: " + str(block.score), False, (0, 0, 0))
        screen.blit(textsurface, (240, 20))
        pop = myfont.render("Population Size: " + str(len(players)), False, (0, 0, 0))
        screen.blit(pop, (20, 150))
        genn = myfont.render("Generation: " + str(gen), False, (0, 0, 0))
        screen.blit(genn, (400, 150))

        pygame.display.update()
        block.move()
        for x, player in enumerate(players):
            if block.collision(player):
                ge[x].fitness -= 1
                players.pop(x)
                nets.pop(x)
                ge.pop(x)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 20)


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
