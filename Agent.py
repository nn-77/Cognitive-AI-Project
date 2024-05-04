"""
Functions to create, train, and obtain predictions from Deep RL agents!

This model is meant to work with ANY environment that has a discrete action space
and a 1-dimensional (Nx1) observation space (such as the RAM state from any Atari environment).
    - to use a higher dimensional observation space (such as the pixels of an image), flatten it to a
    Nx1 vector, and it will work accordingly (not as well as with a CNN policy implementation, but that is not
    the purpose of this file)

PyTorch RL tutorials (from the pytorch site) were used for help with the pure RL portion
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
https://pytorch.org/tutorials/advanced/pendulum.html
https://pytorch.org/tutorials/advanced/coding_ddpg.html
https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

The cognition portion was all made from scratch
"""
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import os
import datetime
from Cognitive_Hypotheses import cognitive_wrapper


MODEL_ONLINE = 1
MODEL_TARGET = 2


# helper to plot the aggregated return arrays
def plot_return(return_file):
    returns = np.load(return_file)
    agg_arr = []
    running = 0
    for i, r in enumerate(returns):
        running += r
        if i != 0 and i % 50 == 0:
            agg_arr.append(running)
            running = 0

    plt.plot([i for i in range(len(agg_arr))], agg_arr)
    plt.title("Total Return Per 50-Episode Group")
    plt.xlabel("Episode Group")
    plt.ylabel("Sum of Returns")
    plt.show()


class DDQN(nn.Module):
    """
        The DDQN neural network - used through the agent wrapper class below
    """
    def __init__(self, observation_dim, action_dim):
        super(DDQN, self).__init__()

        self.online = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.target = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs, model):
        if model == MODEL_ONLINE:
            return self.online(obs)
        elif model == MODEL_TARGET:
            return self.target(obs)

    # def MLP_for_online_and_target(self, observation_dim, action_dim):
    #     return nn.Sequential(
    #         nn.Linear(observation_dim, 512),
    #         nn.ReLU(),
    #         # nn.Dropout(0.2),
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         # nn.Dropout(0.2),
    #         nn.Linear(256, 64),
    #         nn.ReLU(),
    #         # nn.Dropout(0.2),
    #         nn.Linear(64, action_dim),
    #     )
    # def MLP_for_online_and_target(self, observation_dim, action_dim):
    #     return nn.Sequential(
    #         nn.Linear(observation_dim, 256),
    #         nn.ReLU(),
    #         # nn.Dropout(0.2),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         # nn.Dropout(0.2),
    #         nn.Linear(256, action_dim),
    #     )


class NeuralAgent:
    """
    Agent that utilizes a DDQN (Double Deep Q Network) to predict actions

        - Follows the structure of a general wrapper around ML models that I have used in the past - provides functionality
          for saving, loading, training, and evaluating.

        - Essentially a wrapper around dealing with the raw logits from the DDQN, as well as for storing observed samples
          and applying training updates

        - Some of this code and the training loop (for DDQN) guided by pytorch tutorial implementation,
          but here we use a deep MLP instead

    Note that the cognitive improvements will modify the predict method, as that is where
    the "cognition" should come into play

    Only public methods should be 'simulate()', 'render_agent_game()', and 'load()'
    """

    def __init__(self, observation_dim, action_dim, checkpoint_dir,
                 gamma=0.99, lr=0.00004, max_storage_size=30000, batch_size=32, exploration_rate_decay=0.99999975, use_cognition=False):
        """
        Initializes the model and its wrappers for predicting, training, and storing samples

            - observation_dim and action_dim must be integers - used as the input and output # layers in the MLP

            - checkpoint_dir is where the model is periodically saved
        """

        self.use_cognition = use_cognition
        self.cognition_metadata = {}

        # Customizable parameters - will likely vary performance per game as they are changed. Others in this init
        # method will likely not.
        self.gamma = gamma
        self.lr = lr
        self.max_storage_size = max_storage_size
        self.batch_size = batch_size
        self.exploration_rate_decay = exploration_rate_decay

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.checkpoint_dir = checkpoint_dir

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = DDQN(self.observation_dim, self.action_dim).float().to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # see the 'store_training_sample()' method
        self.stored_samples = TensorDictReplayBuffer(storage=LazyMemmapStorage(self.max_storage_size, device=torch.device("cpu")))

        # (from pytorch tutorial - more specific to RL and DDQN specifically)
        self.exploration_rate = 1
        self.exploration_rate_min = 0.03
        self.cur_step = 0
        self.save_every = 5e5
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def __save(self):
        """
            Saves the model and all relevant instance variables to the checkpoint directory specified in the constructor
        """
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        dt = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        torch.save(
            dict(model=self.model.state_dict(), exploration_rate=self.exploration_rate, gamma=self.gamma, lr=self.lr,
                 max_storage_size=self.max_storage_size, batch_size=self.batch_size,
                 exploration_rate_decay=self.exploration_rate_decay, use_cognition=self.use_cognition),
            f"{self.checkpoint_dir}/{dt}_neural_agent_{int(self.cur_step // self.save_every)}.pth",
        )
        print("Saved model!")

    def load(self, file):
        """
            Load the agent (model and relevant instance variables) from a checkpoint file
        """
        checkpoint = torch.load(file)
        self.exploration_rate = checkpoint["exploration_rate"]
        self.model.load_state_dict(checkpoint["model"])

        self.gamma = checkpoint["gamma"]
        self.lr = checkpoint["lr"]
        self.max_storage_size = checkpoint["max_storage_size"]
        self.batch_size = checkpoint["batch_size"]
        self.exploration_rate_decay = checkpoint["exploration_rate_decay"]

        if "use_cognition" in checkpoint and checkpoint["use_cognition"]:
            print("Using Cognition")
            self.use_cognition = checkpoint["use_cognition"]
        else:
            self.use_cognition = False

    def __save_returns(self, returns):
        """
            Helper to save the given array of returns to the checkpoint directory
        """
        returns_prefix = f"{self.checkpoint_dir}/returns"

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.isdir(returns_prefix):
            os.mkdir(returns_prefix)

        np_filename = f"{returns_prefix}/{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.npy"
        np.save(np_filename, returns)

        return np_filename

    def render_agent_game(self, env, steps):
        """
            Step through the environment, using the agent to predict actions
        """
        self.model.eval()
        observation, _ = env.reset()

        for i in range(steps):
            action = self.__predict(observation, env)
            observation, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                observation, _ = env.reset()

        env.close()

    def simulate(self, env, episodes):
        """
            Simulate the environment to train the model progressively, using all the below helpers

                - Assuming env has all the properties of a Gymnasium env
        """
        self.model.train()
        returns = []
        running_episodes_total = 0
        running_loss_total = 0

        for episode in range(episodes):

            episode_return = 0
            observation, _ = env.reset()

            while True:
                action = self.__predict(observation, env)
                next_observation, reward, terminated, truncated, _ = env.step(action)

                self.__store_training_sample(observation, action, reward, next_observation, terminated or truncated)
                loss = self.__train()

                if loss[0] is not None:
                    running_loss_total += loss[1]

                observation = next_observation
                episode_return += reward

                if terminated or truncated:
                    break

            returns.append(episode_return)
            running_episodes_total += episode_return
            if episode % 50 == 0:
                print(f"Episode: {episode}; Exploration rate: {self.exploration_rate}; "
                      f"50-episode return: {running_episodes_total}; 50-episode loss: {running_loss_total};"
                      f" Current step: {self.cur_step}; Elements in memory: {len(self.stored_samples)}")
                running_episodes_total = 0
                running_loss_total = 0

        self.__save()
        self.model.eval()

        return self.__save_returns(returns)

    def __predict(self, observation, env):
        """ Uses an epsilon-greedy algorithm to pick an action

            - (https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/)

            - Either gets a random action or an action decided by the DDQN

            - Much more likely to "explore" and pick random actions at first, and as it gains more distinct experiences,
              it defaults to the DDQN most of the time

            - Ideally would throw an error if predict() is called for inference and the exploration rate is still
              > 0.1 - this means the model has not been trained enough yet

            Cognition is incorporated here as specified in Cognitive_Hypotheses.py

        :param observation: 1D numpy array of length 'self.observation_dim'
        :param env: here to extract the 2D numpy array containing each pixel's rgb values, used for cognition
        :return: action determined by epsilon-greedy algorithm given the observation

        $1: Infer using argmax instead of [probabilistic] softmax because 'self.model' models the Q function, not the policy

        $2: From pytorch tutorial - lower likelihood of picking a non-greedy action as more experiences are gained.
            Caveat is that 'predict()' cannot be called extensively by outside calls until the model has been trained
            a lot through simulate(), otherwise the explored areas will not be stored in 'self.stored_samples' before
            the exploration rate is decayed
        """

        if np.random.rand() < self.exploration_rate:  # epsilon
            if self.use_cognition:
                action = cognitive_wrapper(torch.rand(1, self.action_dim), env.unwrapped.ale.getScreenRGB(), self.cognition_metadata)
            else:
                action = np.random.randint(self.action_dim)

        else:  # greedy
            with torch.no_grad():
                observation = torch.tensor(observation).to(self.device).unsqueeze(0).float()
                logits = self.model(observation, MODEL_ONLINE).cpu()

            if self.use_cognition:
                action = cognitive_wrapper(logits, env.unwrapped.ale.getScreenRGB(), self.cognition_metadata)
            else:
                action = logits.argmax(axis=1).item()  # $1

        # $2
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.cur_step += 1

        return action

    def __store_training_sample(self, observation, action, reward, next_observation, terminated):
        """
            Since we are doing RL, we get training samples (inputs + predictions + labels) as we do inference,
            so we store them all together so that we can do batch updates on the network parameters later
        """

        # (from pytorch tutorials) - TensorDictReplayBuffer is an efficient and good way for storing memory for RL tasks
        observation = torch.tensor(observation).float()
        action = torch.tensor([action])
        reward = torch.tensor([reward]).float()
        next_observation = torch.tensor(next_observation).float()
        terminated = torch.tensor([terminated])
        self.stored_samples.add(TensorDict({
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "terminated": terminated
            }, batch_size=[])
        )

    def __get_training_batch(self):
        """
            Similar to a dataloader for non-RL tasks - get a batch of training samples
        """

        # (from pytorch tutorials) - TensorDictReplayBuffer is an efficient and good way for storing memory for RL tasks
        batch = self.stored_samples.sample(self.batch_size).to(self.device)

        observations, actions, rewards, next_observations, terminated = \
            (batch.get(key) for key in ("observation", "action", "reward", "next_observation", "terminated"))

        return (observations.float(), actions.squeeze(), rewards.squeeze(),
                next_observations.float(), terminated.squeeze().float())

    @torch.no_grad()
    def __td_target(self, rewards, next_observations, terminated):
        """
            Get TD Target part of DDQN theory (from pytorch tutorials)
        """

        next_observation_Q = self.model(next_observations, MODEL_ONLINE)
        optimal_next_action = next_observation_Q.argmax(axis=1)

        next_observation_target_Q = self.model(next_observations, MODEL_TARGET)[
            np.arange(0, self.batch_size), optimal_next_action
        ]

        return (rewards + (1 - terminated.float()) * self.gamma * next_observation_target_Q).float()

    def __train(self):
        """
            Training loop using the above helper methods for DDQN loss (from pytorch tutorial)
        """

        if self.cur_step % self.sync_every == 0:
            # The parameters of the target Q network are never optimized - they are only copied from the online network
            self.model.target.load_state_dict(self.model.online.state_dict())

        if self.cur_step % self.save_every == 0:
            self.__save()

        if self.cur_step < self.burnin:
            return None, None

        if self.cur_step % self.learn_every != 0:
            return None, None

        observations, actions, rewards, next_observations, terminated = self.__get_training_batch()

        td_estimate = self.model(observations, MODEL_ONLINE)[np.arange(0, self.batch_size), actions]
        td_target = self.__td_target(rewards, next_observations, terminated)

        # Perform gradient step using TD values
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True, loss.item()




