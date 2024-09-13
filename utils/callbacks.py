import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from utils.helperFunctions import action_number_into_function,select_features_from_flattened
import matplotlib.pyplot as plt
import wandb

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model that achieves a new best training reward.

    This callback checks the training reward every specified number of steps and saves the model if its performance has improved.
    It is recommended to use this in conjunction with the ``EvalCallback`` for more robust evaluation of the model's performance.

    Parameters
    ----------
    check_freq : int
        How often to check for improvement, in terms of training steps.
    log_dir : str
        Path to the directory where the training logs and model will be saved. This directory must contain the file created by the ``Monitor`` wrapper.
    verbose : int, optional
        Level of verbosity. 0 for no output, 1 for detailed output. Default is 1.

    Attributes
    ----------
    save_path : str
        The path where the best model will be saved.
    best_mean_reward : float
        The highest mean reward achieved by the model so far. Initialized to negative infinity.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        """
        Initialize the callback.

        Parameters
        ----------
        check_freq : int
            How often to check for improvement, in terms of training steps.
        log_dir : str
            Path to the directory where the training logs and model will be saved.
        verbose : int, optional
            Level of verbosity.
        """
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """
        Initialize the callback by creating the directory where the best model will be saved, if it does not already exist.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every step to check if the current step is a check step and, if so, evaluates the model's performance.

        Returns
        -------
        bool
            Always returns True to continue training.
        """
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-10:])  # make this value smaller
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


class ProgressBarCallback(BaseCallback):
    """
    Callback for displaying a progress bar during training.

    Parameters
    ----------
    pbar : tqdm.pbar
        Progress bar object.

    """

    def __init__(self, pbar):
        """
        Initialize the ProgressBarCallback.

        Parameters
        ----------
        pbar : tqdm.pbar
            Progress bar object.
        """
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        """
        Update the progress bar at each step.
        """
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True


class ProgressBarManager(object):
    """
    Manager for the progress bar during training or evaluation.

    This class is designed to be used with a `with` statement, ensuring that the progress bar is properly created and
    destroyed upon completion.

    Parameters
    ----------
    total_timesteps : int
        The total number of timesteps for which the progress bar will track progress.

    Attributes
    ----------
    pbar : tqdm or None
        The progress bar object. Initialized as `None` and set upon entering the context.
    total_timesteps : int
        The total number of timesteps the progress bar will track.
    """

    def __init__(self, total_timesteps):
        """
        Initializes the ProgressBarManager with the total number of timesteps.

        Parameters
        ----------
        total_timesteps : int
            The total number of timesteps for which the progress bar will track progress.
        """
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        """
        Creates and returns a progress bar and its associated callback upon entering the context.

        Returns
        -------
        ProgressBarCallback
            The callback associated with the progress bar to be used during training or evaluation.
        """
        self.pbar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures proper closure of the progress bar upon exiting the context.
        """
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class RudderManager(BaseCallback):
    """
    Callback for creating buffers and then running the rudder algorithm

    Parameters
    sequence_generate: whether or not to create our buffer sequences
    lesson_buffer: the structure to store our sequences
    lstm: our NN that creates our delayed rewards

    """
    def __init__(self, sequence_generate, lesson_buffer,lstm):

        super(RudderManager, self).__init__()
        self.sequence_generate = sequence_generate
        self.lesson_buffer = lesson_buffer
        self.lstm = lstm
        """
        If we want to collect sequences for our LSTM then we need to initialize those data structures
        """
        self.sequence_generate = sequence_generate
        if self.sequence_generate:
            self.seq_state = []
            self.seq_action = []
            self.seq_reward = []
            self.lesson_buffer = lesson_buffer
        self.episode_num = 1


    def _on_step(self):
        if self.sequence_generate:
            self.seq_action.append(self.locals['actions'][0])
            self.seq_state.append(self.locals['new_obs'][0])
            self.seq_reward.append(self.locals['rewards'][0])
            self.mapping = self.model.env.get_attr('flattened_mapping')[0]

            #Save experience to lesson_buffer
            if self.locals['dones'][0]:
                states = np.stack(self.seq_state)
                actions = np.array(self.seq_action)
                rewards = np.array(self.seq_reward)
                print(states.shape, actions.shape, rewards.shape)
                self.lesson_buffer.add(states=states, actions=actions, rewards=rewards)
                if self.lesson_buffer.different_returns_encountered() and self.lesson_buffer.full_enough():

                    print("\nReady to Learn\n")
                    # If RUDDER is run, the LSTM is trained after each episode until its loss is below a threshold.
                    # Samples will be drawn from the lessons buffer.
                    if self.episode_num % 25 == 0:
                        self.lstm.train(episode=self.episode_num,state_mapping=self.mapping)
                        # Then the LSTM is used to redistribute the reward.
                        rewards = self.lstm.redistribute_reward(states=np.expand_dims(states, 0),
                                                       actions=np.expand_dims(actions, 0),state_mapping=self.mapping)[0, :]
                        #For each action we take we want to redistribute that reward to it and combine with the state delta.


                self.seq_action, self.seq_state, self.seq_reward = [], [], []
                self.episode_num += 1

            return True

class EnvLogger(BaseCallback):
    """
    Callback for logging episodes of the environment during training.

    Parameters
    ----------
    log_frequency : int
        How many episodes to wait before logging an episode. (1 -> log every episode, 5 -> log every 5th episode)
    log_dir : str
        Directory where the logs will be saved.

    """

    def __init__(self, log_frequency, log_dir,sequence_generate=False,lesson_buffer=None):
        """
        Initialize the EnvLogger callback.

        Parameters
        ----------
        log_frequency : int
            How many episodes to wait before logging an episode.
        log_dir : str
            Directory where the logs will be saved.
        """
        super(EnvLogger, self).__init__()
        self.log_frequency = log_frequency
        self.log_dir = log_dir

        # create dir if not exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.episode_num = 1

    def _init_callback(self) -> None:
        """
        Initialize the callback by setting up the logging infrastructure.
        """
        """
        We need to represent the data for the environment 
        these are our column headers
        """
        self.tables = self.model.env.get_attr("tables")
        self.immutable_config = self.model.env.get_attr("immutable_config")
        list = []
        for i, table in enumerate(self.tables[0]):
            list.append(f'table_{i}_status')
            list.append(f'party_{i}_status')

        #For coding action #s as actual strings for human readability
        unique_combos = self.model.env.get_attr("unique_combos")
        self.action_dict = action_number_into_function(self.tables[0],unique_combos[0],self.immutable_config[0])
        # Create output frame
        self.df = pd.DataFrame(columns=['action', 'reward','values','log_probs','curr_time'] + list )

    def _on_step(self):
        """
        Called at every step to log the episode data if the current episode number matches the logging frequency.
        """

        if self.episode_num % self.log_frequency == 0:

            # Check done
            if self.locals['dones'][0]:
                self.save_feather()
                self.episode_num += 1
                return True  # Stablebaselines calls reset() before the callback, so this step has invalid values



            # Write action and reward
            row_dict = dict()
            raw_action = self.locals['actions'][0]
            row_dict['action'] =  self.action_dict[raw_action]
            row_dict['reward'] = self.locals['rewards'][0]
            row_dict['values'] = self.locals['values'][0].detach().numpy()[0]
            row_dict['log_probs'] = self.locals['log_probs'].detach().numpy()[0]

            # Show tables
            row_dict['curr_time'] = self.model.env.get_attr("universal_clock")[0].current_time
            for i, table in enumerate(self.model.env.get_attr("tables")[0]):
                row_dict[f'table_{i}_status'] = table.status
                if table.party:
                    row_dict[f'table_{i}_party_status'] = table.party.status
            self.df = pd.concat([self.df, pd.DataFrame([row_dict])], ignore_index=True)

        # Count episodes
        if self.locals['dones'][0]:
            self.episode_num += 1
        return True

    def save_feather(self):
        """
        Save the logged data to a file and reset the logging dataframe.
        """

        # Save to file
        self.df.to_csv(self.log_dir + f"episode_{self.episode_num}.csv", index=False)
        #self.plot_episodic_graphs(path=self.log_dir + f"episode_{self.episode_num}.csv", n_bunkers=len(self.bunkers))
        # Reset logged data
        self.df = self.df[0:0]
        print(f'saved the file for episode_{self.episode_num}')
        return

    def plot_episodic_graphs(self, path=None, n_bunkers=None):
        """
        Plot episodic graphs for the logged data.

        Parameters
        ----------
        path : str, optional
            Path to the CSV file containing the logged data.
        n_bunkers : int, optional
            Number of bunkers to plot data for.
        """
        df_to_plot = pd.read_csv(path).drop(columns=["action"])
        fig, axes = plt.subplots(nrows=n_bunkers + 3, ncols=1, figsize=(15, len(df_to_plot.columns) * 3))
        df_to_plot.plot(subplots=True, ax=axes)
        plt.close()
        wandb.log({f"{self.episode_num}": wandb.Image(fig)})

        print("finished episodic plotting of state variables")