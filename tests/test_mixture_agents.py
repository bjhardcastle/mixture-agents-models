"""
Tests for mixture agents models.

Basic unit tests to validate core functionality of the mixture agents
framework and ensure integration with dynamic routing works correctly.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mixture_agents_models as mam


class TestAgents(unittest.TestCase):
    """Test agent implementations."""
    
    def setUp(self):
        """Set up test data."""
        self.n_trials = 100
        self.choices = np.random.randint(0, 2, self.n_trials)
        self.rewards = np.random.randint(0, 2, self.n_trials)
        self.data = mam.GenericData(
            choices=self.choices,
            rewards=self.rewards,
            n_trials=self.n_trials,
            n_sessions=1
        )
    
    def test_mb_reward_agent(self):
        """Test MBReward agent."""
        agent = mam.MBReward(alpha=0.5)
        
        # Test initial Q values
        self.assertEqual(len(agent.q0), 4)
        self.assertTrue(0 <= agent.alpha <= 1)
        
        # Test Q update
        q = agent.q0.copy()
        q_new = agent.next_q(q, self.data, 0)
        self.assertEqual(len(q_new), len(q))
        self.assertIsInstance(q_new, np.ndarray)
    
    def test_mf_reward_agent(self):
        """Test MFReward agent."""
        agent = mam.MFReward(alpha=0.3)
        
        # Test properties
        self.assertEqual(len(agent.q0), 4)
        self.assertEqual(agent.color, "seagreen")
        
        # Test parameter methods
        params = agent.get_params()
        self.assertIn('alpha', params)
        
        new_agent = agent.set_params(alpha=0.7)
        self.assertEqual(new_agent.alpha, 0.7)
    
    def test_bias_agent(self):
        """Test Bias agent."""
        agent = mam.Bias()
        
        # Bias agent should have zero Q values
        self.assertTrue(np.all(agent.q0 == 0))
        
        # Q values should not change
        q = agent.q0.copy()
        q_new = agent.next_q(q, self.data, 0)
        np.testing.assert_array_equal(q, q_new)
    
    def test_context_rl_agent(self):
        """Test ContextRL agent for dynamic routing."""
        agent = mam.ContextRL(alpha_context=0.6, alpha_reinforcement=0.4)
        
        self.assertEqual(agent.alpha_context, 0.6)
        self.assertEqual(agent.alpha_reinforcement, 0.4)
        
        # Test with context data
        contexts = np.random.randint(0, 2, self.n_trials)
        data_with_context = mam.DynamicRoutingData(
            choices=self.choices,
            rewards=self.rewards,
            n_trials=self.n_trials,
            n_sessions=1,
            contexts=contexts
        )
        
        q = agent.q0.copy()
        q_new = agent.next_q(q, data_with_context, 0)
        self.assertIsInstance(q_new, np.ndarray)


class TestTasks(unittest.TestCase):
    """Test data structures and loading."""
    
    def test_generic_data_creation(self):
        """Test GenericData creation and validation."""
        choices = np.array([0, 1, 1, 0])
        rewards = np.array([0, 1, 0, 1])
        
        data = mam.GenericData(
            choices=choices,
            rewards=rewards,
            n_trials=4,
            n_sessions=1
        )
        
        self.assertEqual(data.n_trials, 4)
        self.assertEqual(data.n_sessions, 1)
        np.testing.assert_array_equal(data.choices, choices)
        np.testing.assert_array_equal(data.rewards, rewards)
    
    def test_data_validation(self):
        """Test data validation catches errors."""
        with self.assertRaises(ValueError):
            # Mismatched lengths
            mam.GenericData(
                choices=np.array([0, 1]),
                rewards=np.array([0, 1, 0]),
                n_trials=2,
                n_sessions=1
            )
    
    def test_dynamic_routing_data(self):
        """Test DynamicRoutingData structure."""
        n_trials = 50
        choices = np.random.randint(0, 2, n_trials)
        rewards = np.random.randint(0, 2, n_trials)
        contexts = np.random.randint(0, 2, n_trials)
        trial_stim = np.array(['vis1' if c == 0 else 'sound1' for c in contexts])
        
        data = mam.DynamicRoutingData(
            choices=choices,
            rewards=rewards,
            n_trials=n_trials,
            n_sessions=1,
            contexts=contexts,
            trial_stim=trial_stim,
            mouse_id="12345",
            session_start_time="2024-01-15"
        )
        
        self.assertEqual(data.mouse_id, "12345")
        self.assertEqual(len(data.trial_stim), n_trials)
        np.testing.assert_array_equal(data.contexts, contexts)


class TestModels(unittest.TestCase):
    """Test model structures and fitting."""
    
    def setUp(self):
        """Set up test data and model."""
        np.random.seed(42)  # For reproducibility
        
        self.n_trials = 200
        self.choices = np.random.randint(0, 2, self.n_trials)
        self.rewards = np.random.randint(0, 2, self.n_trials)
        
        self.data = mam.GenericData(
            choices=self.choices,
            rewards=self.rewards,
            n_trials=self.n_trials,
            n_sessions=1
        )
        
        self.agents = [
            mam.MBReward(alpha=0.5),
            mam.MFReward(alpha=0.4),
            mam.Bias()
        ]
    
    def test_model_hmm_creation(self):
        """Test ModelHMM creation and validation."""
        n_agents, n_states = 3, 2
        
        beta = np.random.normal(0, 1, (n_agents, n_states))
        pi = np.array([0.6, 0.4])
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        model = mam.ModelHMM(beta=beta, pi=pi, A=A)
        
        self.assertEqual(model.beta.shape, (n_agents, n_states))
        self.assertEqual(len(model.pi), n_states)
        self.assertEqual(model.A.shape, (n_states, n_states))
    
    def test_model_options(self):
        """Test ModelOptionsHMM configuration."""
        options = mam.ModelOptionsHMM(
            n_states=2,
            max_iter=50,
            tol=1e-3
        )
        
        self.assertEqual(options.n_states, 2)
        self.assertEqual(options.max_iter, 50)
        self.assertEqual(options.tol, 1e-3)
        
        # Check default initialization
        self.assertIsNotNone(options.pi_0)
        self.assertIsNotNone(options.A_0)
    
    def test_agent_options(self):
        """Test AgentOptions configuration."""
        options = mam.AgentOptions(agents=self.agents)
        
        self.assertEqual(len(options.agents), 3)
        self.assertFalse(options.scale_x)
    
    def test_basic_fitting(self):
        """Test basic model fitting."""
        model_options = mam.ModelOptionsHMM(
            n_states=2, 
            max_iter=10,  # Keep short for testing
            verbose=False
        )
        agent_options = mam.AgentOptions(agents=self.agents)
        
        # This should run without errors
        model, fitted_agents, log_likelihood = mam.optimize(
            data=self.data,
            model_options=model_options,
            agent_options=agent_options,
            verbose=False
        )
        
        self.assertIsInstance(model, mam.ModelHMM)
        self.assertEqual(len(fitted_agents), len(self.agents))
        self.assertIsInstance(log_likelihood, float)
    
    def test_simulation(self):
        """Test model simulation."""
        # Create simple model for testing
        n_agents, n_states = len(self.agents), 2
        beta = np.random.normal(0, 1, (n_agents, n_states))
        pi = np.array([0.5, 0.5])
        A = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        model = mam.ModelHMM(beta=beta, pi=pi, A=A)
        
        predictions = mam.simulate(model, self.agents, self.data, n_reps=2)
        
        self.assertIn('choices', predictions)
        self.assertIn('states', predictions)
        self.assertEqual(predictions['choices'].shape, (2, self.n_trials))


class TestIntegration(unittest.TestCase):
    """Test dynamic routing integration."""
    
    def test_create_dynamic_routing_agents(self):
        """Test agent creation for dynamic routing."""
        agents = mam.create_dynamic_routing_agents()
        
        self.assertGreater(len(agents), 0)
        
        # Check for expected agent types
        agent_types = [type(agent).__name__ for agent in agents]
        self.assertIn('ContextRL', agent_types)
        self.assertIn('Bias', agent_types)
    
    def test_dynamic_routing_data_conversion(self):
        """Test conversion from dynamic routing format."""
        # Create mock session data
        class MockSessionData:
            def __init__(self):
                self.trial_response = np.random.randint(0, 2, 100)
                self.trial_stim = ['vis1' if i % 2 == 0 else 'sound1' for i in range(100)]
                self.auto_reward_scheduled = np.zeros(100, dtype=bool)
                self.rewarded_stim = self.trial_stim.copy()
                self.subject_name = "TestMouse"
                self.start_time = "2024-01-15"
        
        session_data = MockSessionData()
        
        # Test conversion
        data = mam.convert_from_dynamic_routing(session_data)
        
        self.assertIsInstance(data, mam.DynamicRoutingData)
        self.assertEqual(data.n_trials, 100)
        self.assertEqual(data.mouse_id, "TestMouse")
    
    def test_fit_dynamic_routing_model(self):
        """Test fitting model to dynamic routing data."""
        # Create test data
        n_trials = 100
        data = mam.DynamicRoutingData(
            choices=np.random.randint(0, 2, n_trials),
            rewards=np.random.randint(0, 2, n_trials),
            n_trials=n_trials,
            n_sessions=1,
            contexts=np.random.randint(0, 2, n_trials)
        )
        
        # Test fitting (with minimal iterations for speed)
        results = mam.fit_dynamic_routing_model(
            data=data,
            n_states=2,
            max_iter=5,
            verbose=False
        )
        
        self.assertIn('model', results)
        self.assertIn('agents', results)
        self.assertIn('log_likelihood', results)
        self.assertIn('accuracy', results)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_smooth_function(self):
        """Test smoothing function."""
        x = np.array([1, 5, 3, 7, 2])
        smoothed = mam.smooth(x, window_size=3)
        
        self.assertEqual(len(smoothed), len(x))
        # Middle value should be average of neighbors
        expected_middle = np.mean([1, 5, 3])
        self.assertAlmostEqual(smoothed[1], expected_middle, places=5)
    
    def test_onehot_encoding(self):
        """Test one-hot encoding."""
        x = np.array([0, 1, 2, 1])
        onehot = mam.onehot(x, n_classes=3)
        
        self.assertEqual(onehot.shape, (4, 3))
        np.testing.assert_array_equal(onehot[0], [1, 0, 0])
        np.testing.assert_array_equal(onehot[1], [0, 1, 0])
    
    def test_agent_strings(self):
        """Test agent string representation."""
        agents = [mam.MBReward(), mam.Bias()]
        strings = mam.compute_agent_strings(agents)
        
        self.assertEqual(strings, ['MBReward', 'Bias'])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
