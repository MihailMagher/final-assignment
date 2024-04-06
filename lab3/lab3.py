import pygame
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

# Constants
G = 6.674e-11  # N kg-2 m^2
Earth_Mass = 5.972e24  # kg
Moon_Mass = 7.34767309e22  # kg
Distance = 384400000.  # m
#distances = 0

# Set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Clock object that ensures that animation has the same speed on all machines
clock = pygame.time.Clock()

# Load an image
def load_image(name):
    image = pygame.image.load(name)
    return image

class HeavenlyBody(pygame.sprite.Sprite):
    
    def __init__(self, name, mass, color=WHITE, radius=0, imagefile=None):
        pygame.sprite.Sprite.__init__(self)

        if imagefile:
            self.image = load_image(imagefile)
        else:
            self.image = pygame.Surface([radius*2, radius*2])
            self.image.fill(BLACK)
            pygame.draw.circle(self.image, color, (radius, radius), radius, radius)

        self.rect = self.image.get_rect()
        self.pos = np.array([0,0])
        self.vel = np.array([0,0])
        self.mass = mass
        self.radius = radius
        self.name = name
        self.G = G
        self.distances = []
        #change to false if you want to trim the data
        self.record = True

    def set_pos(self, pos):
        self.pos = np.array(pos)

    def set_vel(self, vel):
        self.vel = np.array(vel)

    def update1(self, objects, dt):
        for o in objects:
            if o != self.name:
                other = objects[o]

                d = (other.pos - self.pos)
                r = np.linalg.norm(d)
                self.solver = ode(self.f)
                self.solver.set_integrator('dopri5')
                self.solver.set_initial_value(np.concatenate((self.pos, self.vel)), 0)
                self.solver.set_f_params(objects)
                new_state = self.solver.integrate(self.solver.t + dt)
                newpos = new_state[:2]
                newvel = new_state[2:]

                self.vel = newvel
                self.pos = newpos

                if self.name == 'earth' and self.record:
                    self.distances.append(r)

    def f(self, t, state, objects):
        pos = state[:2]
        vel = state[2:]

        force = np.array([0.0, 0.0])
        for o in objects:
            if o != self.name:
                other = objects[o]

                d = (other.pos - pos)
                r = np.linalg.norm(d)
                u = d / r
                force += u * G * self.mass * other.mass / (r * r)

        return np.concatenate((vel, force / self.mass))

class Universe:
    def __init__(self):
        self.w, self.h = 2.6*Distance, 2.6*Distance 
        self.objects_dict = {}
        self.objects = pygame.sprite.Group()
        self.dt = 10.0

    def add_body(self, body):
        self.objects_dict[body.name] = body
        self.objects.add(body)

    def to_screen(self, pos):
        return [int((pos[0] + 1.3*Distance)*640//self.w), int((pos[1] + 1.3*Distance)*640.//self.h)]

    def update(self):
        for o in self.objects_dict:
            obj = self.objects_dict[o]
            obj.update1(self.objects_dict, self.dt)
            p = self.to_screen(obj.pos)

            obj.rect.x, obj.rect.y = p[0]-obj.radius, p[1]-obj.radius
        self.objects.update()

    def draw(self, screen):
        self.objects.draw(screen)

def main():
    strikes = input("how many strikes do you want to have? type -1 for infinite, 30 recomended: ")
    force = input("with how much force do you want to strike? 250 is the default: ")
    force = int(force)
    strikes=int(strikes)
    maxstrikes=strikes
    textresponse2 = "Impart force on the moon using wasd, press u and j for more or less force"

    print ('Press q to quit')

    # Initializing pygame
    pygame.init()
    win_width = 640
    win_height = 640
    screen = pygame.display.set_mode((win_width, win_height))  # Top left corner is (0,0)
    pygame.display.set_caption('Heavenly Bodies')

    # Create a Universe object, which will hold our heavenly bodies (planets, stars, moons, etc.)
    universe = Universe()

    earth = HeavenlyBody('earth', Earth_Mass, radius=32, imagefile='earth-northpole.jpg')
    earth.set_pos([0, 0])
    moon = HeavenlyBody('moon', Moon_Mass, WHITE, radius=10)
    moon.set_pos([Distance, 0])
    moon.set_vel([0, 1024.5])

    universe.add_body(earth)
    universe.add_body(moon)

    font = pygame.font.Font(None, 24)
    font2=pygame.font.Font(None,54)
    text = font.render("strikes left:", True, WHITE)

    total_frames = 1000000
    iter_per_frame = 150

    frame = 0
    ended =0
    while True:
        d = np.linalg.norm(moon.pos - earth.pos)
        v=np.linalg.norm(moon.vel)
        if False:
            print ('Frame number', frame) 
        if strikes>0:
            strikeAmountText=font2.render(str(strikes), True, WHITE)

        else:
            strikeAmountText=font2.render("N/A", True, WHITE)
        text2 = font.render(textresponse2, True, WHITE)
        dtext = font.render(("Distance: "+str(d)),True,WHITE)
        vtext = font.render(("Velocity: "+str(v)),True,WHITE)
        ftext = font.render(("force: "+str(force)),True,WHITE)
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            ended=1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            ended=1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_s and strikes != 0:
            moon.vel[1] += int(force)
            strikes-=1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_w and strikes != 0:
            moon.vel[1] -= int(force)
            strikes-=1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_a and strikes != 0:
            moon.vel[0] -= int(force)
            strikes-=1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_d and strikes != 0:
            moon.vel[0] += int(force)
            strikes-=1
        else:
            pass
        if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
            force +=50
        elif(event.type==pygame.KEYDOWN and event.key == pygame.K_j):
            force-=50
        #if event.type == pygame.KEYDOWN and event.key == pygame.K_k:
            #earth.record=True
            #textresponse2="recording..."
        if d>650000000 or d<60000000:
            ended=1
            print("ERROR:ended due to unstable orbit")   
        universe.update()
        if frame % iter_per_frame == 0:
            screen.fill(BLACK) # clear the background
            universe.draw(screen)
            screen.blit(text, (win_width - 100, win_height - 100))
            screen.blit(strikeAmountText, (win_width - 65, win_height - 60))
            screen.blit(text2,(30,30))
            screen.blit(dtext,(30,win_height-45))
            screen.blit(vtext,(30,win_height-15))
            screen.blit(ftext,(30,win_height-75))
            pygame.display.flip()
        frame += 1
        if ended==1:
            pygame.quit()
            sys.exit
            break
    maxes =[]
    mins=[]
    for i in range(len(earth.distances)-2):
        if earth.distances[i+1]>earth.distances[i] and earth.distances[i+1]> earth.distances[i+2]:
            maxes.append(earth.distances[i+1])
    for i in range(len(earth.distances)-2):
        if earth.distances[i+1]<earth.distances[i] and earth.distances[i+1]< earth.distances[i+2]:
            mins.append(earth.distances[i+1])

    print("there were ",len(maxes)," peaks over the simulation period with ", maxstrikes - strikes," total strikes")
    print("the list of all peaks is: ", maxes)
    print("the list of all troughs is: ", mins)
    plt.figure(1)
    plt.plot(earth.distances)
    plt.xlabel('frame')
    plt.ylabel('distance')
    plt.title('Distance between the earth and the moon')
    plt.show()
            
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
