"""
Data structures and utilities for behavioral experiments.

This module provides data containers and loading functions for 
various experimental paradigms including two-step tasks, 
dynamic routing, and generic behavioral experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io


@dataclass
class GenericData:
    """
    Generic behavioral data structure for choice experiments.
    
    Provides a standardized interface for behavioral data across
    different experimental paradigms.
    """
    
    choices: np.ndarray
    rewards: np.ndarray
    n_trials: int
    n_sessions: int
    session_indices: Optional[np.ndarray] = None
    trial_types: Optional[np.ndarray] = None
    contexts: Optional[np.ndarray] = None
    stimuli: Optional[np.ndarray] = None
    reaction_times: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate data consistency after initialization."""
        if len(self.choices) != len(self.rewards):
            raise ValueError("Choices and rewards must have same length")
        
        if len(self.choices) != self.n_trials:
            raise ValueError("Number of trials must match data length")
        
        # Convert to numpy arrays if needed
        self.choices = np.asarray(self.choices)
        self.rewards = np.asarray(self.rewards)
        
        if self.session_indices is None:
            self.session_indices = np.zeros(self.n_trials, dtype=int)
    
    def get_session_data(self, session_idx: int) -> 'GenericData':
        """Extract data for a specific session."""
        mask = self.session_indices == session_idx
        
        return GenericData(
            choices=self.choices[mask],
            rewards=self.rewards[mask],
            n_trials=int(mask.sum()),
            n_sessions=1,
            session_indices=np.zeros(int(mask.sum()), dtype=int),
            trial_types=self.trial_types[mask] if self.trial_types is not None else None,
            contexts=self.contexts[mask] if self.contexts is not None else None,
            stimuli=self.stimuli[mask] if self.stimuli is not None else None,
            reaction_times=self.reaction_times[mask] if self.reaction_times is not None else None,
            metadata=self.metadata.copy()
        )


@dataclass
class TwoStepData(GenericData):
    """
    Data structure for two-step reinforcement learning tasks.
    
    Extends GenericData with second-stage choices and rewards
    specific to two-step paradigms.
    """
    
    choices_2: np.ndarray = field(default_factory=lambda: np.array([]))
    rewards_2: np.ndarray = field(default_factory=lambda: np.array([]))
    states_2: np.ndarray = field(default_factory=lambda: np.array([]))
    transitions: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self) -> None:
        """Validate two-step specific data."""
        super().__post_init__()
        
        # Ensure second stage data consistency
        if len(self.choices_2) > 0:
            for arr in [self.choices_2, self.rewards_2, self.states_2, self.transitions]:
                if len(arr) != self.n_trials:
                    raise ValueError("Second stage data must match trial count")


@dataclass  
class DynamicRoutingData(GenericData):
    """
    Data structure for dynamic routing behavioral experiments.
    
    Specialized for multi-modal decision tasks with context switches
    and integration with existing dynamic routing analysis pipeline.
    """
    
    trial_stim: np.ndarray = field(default_factory=lambda: np.array([]))
    trial_block: np.ndarray = field(default_factory=lambda: np.array([]))
    trial_opto_label: np.ndarray = field(default_factory=lambda: np.array([]))
    auto_reward_scheduled: np.ndarray = field(default_factory=lambda: np.array([]))
    rewarded_stim: np.ndarray = field(default_factory=lambda: np.array([]))
    stim_start_times: np.ndarray = field(default_factory=lambda: np.array([]))
    mouse_id: Optional[str] = None
    session_start_time: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate dynamic routing specific data."""
        super().__post_init__()
        
        # Ensure all dynamic routing arrays match trial count
        for attr_name in ['trial_stim', 'trial_block', 'trial_opto_label', 
                         'auto_reward_scheduled', 'rewarded_stim', 'stim_start_times']:
            arr = getattr(self, attr_name)
            if len(arr) > 0 and len(arr) != self.n_trials:
                raise ValueError(f"{attr_name} must match trial count")


def load_generic_data(file_path: Union[str, Path], **kwargs: Any) -> GenericData:
    """
    Load behavioral data from various file formats.
    
    Args:
        file_path: Path to data file (.csv, .mat, .npz supported)
        **kwargs: Additional arguments passed to specific loaders
        
    Returns:
        GenericData object with loaded behavioral data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return _load_csv_data(file_path, **kwargs)
    elif file_path.suffix == '.mat':
        return _load_mat_data(file_path, **kwargs)
    elif file_path.suffix == '.npz':
        return _load_npz_data(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _load_csv_data(file_path: Path, **kwargs: Any) -> GenericData:
    """Load data from CSV file."""
    df = pd.read_csv(file_path, **kwargs)
    
    # Required columns
    choices = df['choices'].values
    rewards = df['rewards'].values
    
    # Optional columns
    session_indices = df.get('session', np.zeros(len(choices))).values
    trial_types = df.get('trial_type', None)
    contexts = df.get('context', None)
    stimuli = df.get('stimulus', None)
    reaction_times = df.get('rt', None)
    
    # Convert optional arrays to numpy arrays if they exist
    for arr_name, arr in [('trial_types', trial_types), ('contexts', contexts), 
                         ('stimuli', stimuli), ('reaction_times', reaction_times)]:
        if arr is not None:
            arr = np.asarray(arr.values if hasattr(arr, 'values') else arr)
    
    return GenericData(
        choices=choices,
        rewards=rewards,
        n_trials=len(choices),
        n_sessions=int(session_indices.max()) + 1 if len(session_indices) > 0 else 1,
        session_indices=session_indices.astype(int),
        trial_types=trial_types,
        contexts=contexts,
        stimuli=stimuli,
        reaction_times=reaction_times,
        metadata={'source_file': str(file_path)}
    )


def _load_mat_data(file_path: Path, **kwargs: Any) -> GenericData:
    """Load data from MATLAB .mat file."""
    data = scipy.io.loadmat(file_path, **kwargs)
    
    # Extract required fields (adjust field names as needed)
    choices = np.squeeze(data.get('choices', data.get('choice', [])))
    rewards = np.squeeze(data.get('rewards', data.get('reward', [])))
    
    # Extract optional fields
    session_indices = np.squeeze(data.get('session', np.zeros(len(choices))))
    trial_types = data.get('trial_type', None)
    contexts = data.get('context', None)
    stimuli = data.get('stimulus', None)
    reaction_times = data.get('rt', None)
    
    return GenericData(
        choices=choices,
        rewards=rewards,
        n_trials=len(choices),
        n_sessions=int(session_indices.max()) + 1 if len(session_indices) > 0 else 1,
        session_indices=session_indices.astype(int),
        trial_types=np.squeeze(trial_types) if trial_types is not None else None,
        contexts=np.squeeze(contexts) if contexts is not None else None,
        stimuli=np.squeeze(stimuli) if stimuli is not None else None,
        reaction_times=np.squeeze(reaction_times) if reaction_times is not None else None,
        metadata={'source_file': str(file_path)}
    )


def _load_npz_data(file_path: Path, **kwargs: Any) -> GenericData:
    """Load data from numpy .npz file."""
    data = np.load(file_path, **kwargs)
    
    choices = data['choices']
    rewards = data['rewards']
    
    return GenericData(
        choices=choices,
        rewards=rewards,
        n_trials=len(choices),
        n_sessions=int(data.get('n_sessions', 1)),
        session_indices=data.get('session_indices', np.zeros(len(choices), dtype=int)),
        trial_types=data.get('trial_types', None),
        contexts=data.get('contexts', None),
        stimuli=data.get('stimuli', None),
        reaction_times=data.get('reaction_times', None),
        metadata={'source_file': str(file_path)}
    )


def load_twostep_data(file_path: Union[str, Path], rat_id: Optional[int] = None, **kwargs: Any) -> TwoStepData:
    """
    Load two-step task data from MATLAB file.
    
    Args:
        file_path: Path to .mat file containing two-step data
        rat_id: Specific rat to load (if file contains multiple rats)
        **kwargs: Additional arguments for scipy.io.loadmat
        
    Returns:
        TwoStepData object with two-step behavioral data
    """
    file_path = Path(file_path)
    data = scipy.io.loadmat(file_path, **kwargs)
    
    # Implementation depends on specific data structure
    # This is a template that should be adapted to actual file format
    
    if rat_id is not None:
        # Extract data for specific rat
        rat_data = data[f'rat_{rat_id}'] if f'rat_{rat_id}' in data else data['data'][rat_id]
    else:
        rat_data = data
    
    # Extract first stage data
    choices = np.squeeze(rat_data['choices'])
    rewards = np.squeeze(rat_data['rewards'])
    
    # Extract second stage data
    choices_2 = np.squeeze(rat_data.get('choices_2', []))
    rewards_2 = np.squeeze(rat_data.get('rewards_2', []))
    states_2 = np.squeeze(rat_data.get('states_2', []))
    transitions = np.squeeze(rat_data.get('transitions', []))
    
    return TwoStepData(
        choices=choices,
        rewards=rewards,
        choices_2=choices_2,
        rewards_2=rewards_2,
        states_2=states_2,
        transitions=transitions,
        n_trials=len(choices),
        n_sessions=1,
        metadata={'source_file': str(file_path), 'rat_id': rat_id}
    )
