import numpy as np
import torch
import torchvision.transforms as T

from environment import Environment


class EnvironmentManager:
    def __init__(self, env: Environment, device):
        self.env = env
        self.device = device
        self.done = False
        self.current_screen = None

    def reset(self):
        self.env.reset()
        return self.get_state()

    def render(self):
        frame = self.env.render(mode="training")
        return frame

    def num_actions_available(self):
        return len(self.env.actions)

    def take_action(self, action):
        reward, self.done = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            self.current_screen = self.get_processed_screen()
        return self.current_screen

    def get_score(self):
        return self.env.score

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[1]

    def get_processed_screen(self):
        screen = self.render()
        return self.transform_screen_data(screen)

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        resize = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])
        return resize(screen).unsqueeze(0).to(self.device)
