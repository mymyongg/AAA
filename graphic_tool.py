import pygame
import numpy as np

BLACK= (0, 0, 0)
WHITE= (255,255,255)
BLUE = (0, 0,255)
GREEN= (0,255, 0)
RED  = (255, 0, 0)

class GraphicTool():
    def __init__(self):
        pygame.init()
        self.size = [400, 400]
        # self.size = [1000, 1000]
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Graphical Plot")
        self.clock = pygame.time.Clock()

    def draw_map(self, obs):
        self.screen.fill((0, 0, 0))
        center_pts = np.array([[-obs['L0'][1], 0.0],
                               [-obs['L2'][1], 2.0],
                               [-obs['L4'][1], 4.0],
                               [-obs['L6'][1], 6.0],
                               [-obs['L8'][1], 8.0],
                               [-obs['L10'][1], 10.0]])
        left_pts = center_pts - [1.75, 0.0]
        right_pts = center_pts + [1.75, 0.0]
        car_loc = np.array([0.0, 0.0])

        center_pts = ((center_pts + [obs['L0'][1], 0.0])) * [100.0, -100.0] + [500.0, 1000.0]
        left_pts = ((left_pts + [obs['L0'][1], 0.0])) * [100.0, -100.0] + [500.0, 1000.0]
        right_pts = ((right_pts + [obs['L0'][1], 0.0])) * [100.0, -100.0] + [500.0, 1000.0]
        car_loc = ((car_loc + [obs['L0'][1], 0.0])) * [100.0, -100.0] + [500.0, 1000.0]

        pygame.draw.lines(self.screen, BLUE, False, center_pts)
        pygame.draw.lines(self.screen, WHITE, False, left_pts)
        pygame.draw.lines(self.screen, WHITE, False, right_pts)
        pygame.draw.lines(self.screen, GREEN, True, [car_loc+[90, 160], car_loc+[-90, 160], car_loc+[-90, -160], car_loc+[90, -160]])

        pygame.display.flip()
    
    def update_plot(self, true_y, est_y, true_psi, est_psi, y4_unc, y6_unc, y8_unc, y10_unc):
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (255, 255, 255), [100, 0], [100, 400], 2) # Left line
        pygame.draw.line(self.screen, (255, 255, 255), [300, 0], [300, 400], 2) # Right line
        pygame.draw.line(self.screen, (255, 255, 0), [200, 0], [200, 400], 2) # Center line
        
        true_center = np.array([200 + true_y*200/3.5, 200])
        est_center = np.array([200 + est_y*200/3.5, 200])
        p1_true, p2_true, p3_true, p4_true = self.rotate(true_psi)
        p1_est, p2_est, p3_est, p4_est = self.rotate(est_psi)
        p1_true = list(p1_true + true_center)
        p2_true = list(p2_true + true_center)
        p3_true = list(p3_true + true_center)
        p4_true = list(p4_true + true_center)
        p1_est  = list(p1_est + est_center)
        p2_est  = list(p2_est + est_center)
        p3_est  = list(p3_est + est_center)
        p4_est  = list(p4_est + est_center)
        
        # vehicle box
        pygame.draw.polygon(self.screen, (0, 0, 255), [p1_true, p2_true, p3_true, p4_true], 2)
        # pygame.draw.polygon(self.screen, (0, 255, 0), [p1_est, p2_est, p3_est, p4_est], 2)

        # # Uncertainty
        # pygame.draw.rect(self.screen, (255, 0, 0), [310, 160, 80, 20])
        # pygame.draw.rect(self.screen, (0, 255, 0), [310, 160, y4_unc / 0.1 * 80, 20])
        
        # pygame.draw.rect(self.screen, (255, 0, 0), [310, 190, 80, 20])
        # pygame.draw.rect(self.screen, (0, 255, 0), [310, 190, y6_unc / 0.1 * 80, 20])
        
        # pygame.draw.rect(self.screen, (255, 0, 0), [310, 220, 80, 20])
        # pygame.draw.rect(self.screen, (0, 255, 0), [310, 220, y8_unc / 0.1 * 80, 20])

        # pygame.draw.rect(self.screen, (255, 0, 0), [310, 250, 80, 20])
        # pygame.draw.rect(self.screen, (0, 255, 0), [310, 250, y10_unc / 0.1 * 80, 20])

        pygame.display.flip()

    def rotate(self, theta):
        p1 = np.array([[40],
                       [80]])
        p2 = np.array([[-40],
                       [80]])
        p3 = np.array([[-40],
                       [-80]])
        p4 = np.array([[40],
                       [-80]])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        p1 = np.matmul(rot_mat, p1)[:, 0]
        p2 = np.matmul(rot_mat, p2)[:, 0]
        p3 = np.matmul(rot_mat, p3)[:, 0]
        p4 = np.matmul(rot_mat, p4)[:, 0]
        return p1, p2, p3, p4
        
    
    def close_window(self):
        pygame.quit()
