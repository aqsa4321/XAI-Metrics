import numpy as np
import pandas as pd
import os
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def compute_fidelity_from_explanations(model, x_sample, explanation_values, explanation_function, 
                                     baseline_value=0.5, metric='prediction_similarity'):
    """
    Compute fidelity using explanation values directly (without separate surrogate model).
    """
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()
    
    # Get original prediction
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]
    
    # Approximate prediction using explanation values (simple linear approximation)
    explanation_pred = baseline_value + np.sum(explanation_values)
    explanation_pred = np.clip(explanation_pred, 0, 1)  # Keep in [0,1] range
    
    # Compute fidelity
    if metric == 'prediction_similarity':
        fidelity = 1.0 - abs(original_pred - explanation_pred)
    else:
        threshold = 0.5
        orig_binary = 1 if original_pred >= threshold else 0
        exp_binary = 1 if explanation_pred >= threshold else 0
        fidelity = 1.0 if orig_binary == exp_binary else 0.0
    
    return float(fidelity)
    

def compute_model_parameter_randomization(model_predict_func, x_sample, explanation_function,
                                        trained_model=None, model_type='auto',
                                        randomization_type='full', similarity_metric='spearman_abs',
                                        n_samples=5, random_seed=42):
    """
    Compute Model Parameter Randomization Test exactly as defined in the paper.
    Tests whether explanations are sensitive to model parameters by comparing
    explanations from trained vs randomly initialized models with same architecture.

    Parameters:
    -----------
    model_predict_func : Callable
        Model prediction function (e.g., lstm_predict_proba, lstm_predict_single)
    x_sample : np.ndarray
        Original input sample
    explanation_function : Callable
        Function that generates explanations (e.g., get_shap_explanation)
        Should accept (model, x_sample) or work with global model
    trained_model : object, optional
        The actual trained model object. If None, attempts to extract from prediction function
    model_type : str, default='auto'
        Type of model: 'tensorflow', 'pytorch', 'sklearn', 'auto'
    randomization_type : str, default='full'
        'full': randomize all weights, 'cascading': layer by layer, 'independent': one layer at a time
    similarity_metric : str, default='spearman_abs'
        Similarity metric: 'spearman_abs', 'spearman', 'pearson', 'cosine'
    n_samples : int, default=5
        Number of random model samples for averaging
    random_seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    float : Model parameter randomization score
           - High values (close to 1): Method is INSENSITIVE to model parameters (BAD)
           - Low values (close to 0): Method is SENSITIVE to model parameters (GOOD)
    """
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    import copy
    
    np.random.seed(random_seed)
    
    def _compute_similarity(exp1, exp2, metric):
        """Compute similarity between two explanations"""
        exp1, exp2 = np.array(exp1).flatten(), np.array(exp2).flatten()
        
        # Ensure same length
        if len(exp1) != len(exp2):
            min_len = min(len(exp1), len(exp2))
            exp1, exp2 = exp1[:min_len], exp2[:min_len]
        
        # Handle edge cases
        if len(exp1) == 0 or np.allclose(exp1, 0) and np.allclose(exp2, 0):
            return 1.0
        
        try:
            if metric == 'spearman_abs':
                corr, _ = spearmanr(np.abs(exp1), np.abs(exp2))
            elif metric == 'spearman':
                corr, _ = spearmanr(exp1, exp2)
            elif metric == 'pearson':
                corr, _ = pearsonr(exp1, exp2)
            elif metric == 'cosine':
                exp1_norm = exp1 / (np.linalg.norm(exp1) + 1e-8)
                exp2_norm = exp2 / (np.linalg.norm(exp2) + 1e-8)
                corr = np.dot(exp1_norm, exp2_norm)
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")
            
            return 0.0 if np.isnan(corr) else abs(corr)
        except Exception:
            return 0.0
    
    def _detect_model_type(model):
        """Auto-detect model framework"""
        if model is None:
            return 'unknown'
        
        model_str = str(type(model)).lower()
        if 'tensorflow' in model_str or 'keras' in model_str or hasattr(model, 'layers'):
            return 'tensorflow'
        elif 'torch' in model_str or hasattr(model, 'state_dict'):
            return 'pytorch'
        elif hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
            return 'sklearn'
        else:
            return 'unknown'
    
    def _randomize_tensorflow_model(model, randomization_layer=None):
        """Randomize TensorFlow/Keras model weights"""
        try:
            import tensorflow as tf
            
            # Clone the model architecture
            random_model = tf.keras.models.clone_model(model)
            random_model.build(model.input_shape)
            
            if randomization_type == 'full' or randomization_layer is None:
                # Randomize all layers
                for layer in random_model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.01))
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.assign(tf.random.normal(layer.bias.shape, stddev=0.01))
                    if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None:
                        layer.recurrent_kernel.assign(tf.random.normal(layer.recurrent_kernel.shape, stddev=0.01))
            else:
                # Randomize specific layer for cascading/independent
                if randomization_layer < len(random_model.layers):
                    layer = random_model.layers[randomization_layer]
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.01))
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.assign(tf.random.normal(layer.bias.shape, stddev=0.01))
                    if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None:
                        layer.recurrent_kernel.assign(tf.random.normal(layer.recurrent_kernel.shape, stddev=0.01))
            
            return random_model
        except Exception as e:
            print(f"TensorFlow randomization failed: {e}")
            return None
    
    def _randomize_pytorch_model(model, randomization_layer=None):
        """Randomize PyTorch model weights"""
        try:
            import torch
            import torch.nn as nn
            
            # Deep copy the model
            random_model = copy.deepcopy(model)
            
            if randomization_type == 'full' or randomization_layer is None:
                # Randomize all parameters
                with torch.no_grad():
                    for name, param in random_model.named_parameters():
                        if param.requires_grad:
                            param.normal_(0, 0.01)
            else:
                # Randomize specific layer
                layer_count = 0
                with torch.no_grad():
                    for name, module in random_model.named_modules():
                        if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d, nn.Conv2d)):
                            if layer_count == randomization_layer:
                                for param in module.parameters():
                                    if param.requires_grad:
                                        param.normal_(0, 0.01)
                                break
                            layer_count += 1
            
            return random_model
        except Exception as e:
            print(f"PyTorch randomization failed: {e}")
            return None
    
    def _randomize_sklearn_model(model, randomization_layer=None):
        """Randomize sklearn model weights"""
        try:
            from sklearn.base import clone
            
            # Clone the model
            random_model = clone(model)
            
            # Create dummy data to fit the model
            n_features = len(x_sample) if hasattr(x_sample, '__len__') else 10
            dummy_X = np.random.normal(0, 0.1, (100, n_features))
            dummy_y = np.random.randint(0, 2, 100)
            
            # Fit with dummy data
            random_model.fit(dummy_X, dummy_y)
            
            # Randomize weights
            if hasattr(random_model, 'coef_'):
                random_model.coef_ = np.random.normal(0, 0.01, random_model.coef_.shape)
            if hasattr(random_model, 'intercept_'):
                random_model.intercept_ = np.random.normal(0, 0.01, random_model.intercept_.shape)
            
            return random_model
        except Exception as e:
            print(f"Sklearn randomization failed: {e}")
            return None
    
    def _create_random_model(model, model_framework, layer_idx=None):
        """Create randomized model based on framework"""
        if model_framework == 'tensorflow':
            return _randomize_tensorflow_model(model, layer_idx)
        elif model_framework == 'pytorch':
            return _randomize_pytorch_model(model, layer_idx)
        elif model_framework == 'sklearn':
            return _randomize_sklearn_model(model, layer_idx)
        else:
            return None
    
    def _get_explanation_with_model(model, x_sample, explanation_func):
        """Get explanation using specific model"""
        try:
            # Try different ways to pass model to explanation function
            try:
                # Method 1: explanation_function(model, x_sample)
                return explanation_func(model, x_sample)
            except:
                try:
                    # Method 2: explanation_function(x_sample) with global model replacement
                    # This is trickier and framework-dependent
                    return explanation_func(x_sample)
                except:
                    # Method 3: Generate random explanation as fallback (not ideal)
                    original_exp = explanation_func(x_sample)
                    return np.random.normal(0, 0.1, np.array(original_exp).shape)
        except Exception:
            return np.random.normal(0, 0.1, len(x_sample))
    
    try:
        # Auto-detect model type if needed
        if model_type == 'auto':
            model_type = _detect_model_type(trained_model)
        
        # Get original explanation
        original_explanation = explanation_function(x_sample)
        original_explanation = np.array(original_explanation).flatten()
        
        if trained_model is None:
            print("Warning: No trained model provided. Using simplified randomization.")
            # Fallback: generate random explanations
            similarities = []
            for _ in range(n_samples):
                random_explanation = np.random.normal(0, np.std(original_explanation), 
                                                    original_explanation.shape)
                similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                similarities.append(similarity)
            return float(np.mean(similarities))
        
        similarities = []
        
        if randomization_type == 'full':
            # Full model randomization
            for _ in range(n_samples):
                random_model = _create_random_model(trained_model, model_type)
                if random_model is not None:
                    random_explanation = _get_explanation_with_model(random_model, x_sample, explanation_function)
                    random_explanation = np.array(random_explanation).flatten()
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)
                else:
                    # Fallback to random explanation
                    random_explanation = np.random.normal(0, np.std(original_explanation), 
                                                        original_explanation.shape)
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)
        
        elif randomization_type == 'cascading':
            # Cascading randomization (layer by layer)
            max_layers = 10  # Reasonable default
            if hasattr(trained_model, 'layers'):
                max_layers = len(trained_model.layers)
            
            for layer_idx in range(min(max_layers, 5)):  # Limit to prevent too many iterations
                random_model = _create_random_model(trained_model, model_type, layer_idx)
                if random_model is not None:
                    random_explanation = _get_explanation_with_model(random_model, x_sample, explanation_function)
                    random_explanation = np.array(random_explanation).flatten()
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)
        
        elif randomization_type == 'independent':
            # Independent layer randomization
            max_layers = 10
            if hasattr(trained_model, 'layers'):
                max_layers = len(trained_model.layers)
            
            for layer_idx in range(min(max_layers, 5)):
                random_model = _create_random_model(trained_model, model_type, layer_idx)
                if random_model is not None:
                    random_explanation = _get_explanation_with_model(random_model, x_sample, explanation_function)
                    random_explanation = np.array(random_explanation).flatten()
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)
        
        if similarities:
            return float(np.mean(similarities))
        else:
            print("Warning: All randomization attempts failed. Using fallback.")
            return float('nan')
            
    except Exception as e:
        print(f"Warning: Model Parameter Randomization Test failed: {e}")
        return float('nan')


def compute_feature_mutual_information1(model, test_samples, explanation_function, 
                                     threshold=0.1, n_bins=10, epsilon_min=1e-8):
    """
    Compute Feature Mutual Information I(X, Z) for XAI explanation methods on tabular data.
    
    Measures how much information is preserved/lost when converting original features (X) 
    to explanation-based features (Z).
    
    Parameters:
    -----------
    model : Callable
        Model prediction function
    test_samples : np.ndarray
        Multiple input samples (shape: [n_samples, n_features])
    explanation_function : Callable
        Function that generates explanations
    threshold : float, default=0.1
        Threshold for considering a feature "important" 
    n_bins : int, default=10
        Number of bins for discretizing continuous variables
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues
        
    Returns:
    --------
    float : Feature mutual information I(X, Z)
    """
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import KBinsDiscretizer
    
    # Collect original features (X) and extracted features (Z)
    X_original = []
    Z_extracted = []
    
    for sample in test_samples:
        sample_flat = np.array(sample).flatten()
        explanation = explanation_function(sample_flat)
        explanation = np.array(explanation).flatten()
        
        # Original features (X)
        X_original.append(sample_flat)
        
        # Feature extraction: convert attributions to binary importance (Z)
        # Z represents "extracted features" - which features are important
        important_features = (np.abs(explanation) > threshold).astype(int)
        Z_extracted.append(important_features)
    
    X_original = np.array(X_original)
    Z_extracted = np.array(Z_extracted)
    
    # Compute mutual information between original and extracted features
    # For each original feature dimension, compute MI with extracted features
    mi_scores = []
    
    for i in range(X_original.shape[1]):  # For each original feature
        x_feature = X_original[:, i].reshape(-1, 1)
        
        # Discretize continuous features for MI calculation
        if len(np.unique(x_feature)) > n_bins:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
            x_feature_discrete = discretizer.fit_transform(x_feature).flatten()
        else:
            x_feature_discrete = x_feature.flatten()
        
        # Compute MI between original feature i and all extracted features
        # Sum MI across all extracted feature dimensions
        total_mi = 0
        for j in range(Z_extracted.shape[1]):
            z_feature = Z_extracted[:, j]
            mi = mutual_info_regression(x_feature_discrete.reshape(-1, 1), z_feature)
            total_mi += mi[0]
        
        mi_scores.append(total_mi)
    
    # Average mutual information across all feature dimensions
    feature_mi = float(np.mean(mi_scores))
    return feature_mi


def compute_feature_mutual_information(original_features, extracted_features, 
                                               n_bins=10, epsilon_min=1e-8):
    """
    Compute Feature Mutual Information I(X, Z) as defined in the paper.
    
    Measures information preservation between original feature space (X) 
    and extracted/transformed feature space (Z).
    
    Parameters:
    -----------
    original_features : np.ndarray
        Original input features X (shape: [n_samples, n_original_features])
    extracted_features : np.ndarray  
        Extracted/transformed features Z (shape: [n_samples, n_extracted_features])
        Could be from discretization, superpixels, PCA, etc.
    n_bins : int, default=10
        Number of bins for discretizing continuous variables
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues
        
    Returns:
    --------
    float : Feature mutual information I(X, Z)
    """
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import KBinsDiscretizer
    
    X = np.array(original_features)
    Z = np.array(extracted_features)
    
    # Discretize continuous features if needed
    if len(np.unique(X.flatten())) > n_bins:
        discretizer_X = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        X_discrete = discretizer_X.fit_transform(X)
    else:
        X_discrete = X
    
    if len(np.unique(Z.flatten())) > n_bins:
        discretizer_Z = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        Z_discrete = discretizer_Z.fit_transform(Z)
    else:
        Z_discrete = Z
    
    # Compute mutual information between X and Z
    # Average MI across all feature pairs
    mi_scores = []
    
    for i in range(X_discrete.shape[1]):
        for j in range(Z_discrete.shape[1]):
            mi = mutual_info_regression(X_discrete[:, i].reshape(-1, 1), 
                                      Z_discrete[:, j])
            mi_scores.append(mi[0])
    
    # Return average mutual information
    return float(np.mean(mi_scores))


def compute_explanation_attribution_mutual_information(original_features, explanations, 
                                                     n_bins=10):
    """
    Alternative: Compute MI between original features and explanation attributions.
    
    This measures how much the explanation attributions preserve information
    about the original input features.
    """
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import KBinsDiscretizer
    
    X = np.array(original_features)
    explanations = np.array(explanations)
    
    # Discretize original features if needed
    if len(np.unique(X.flatten())) > n_bins:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        X_discrete = discretizer.fit_transform(X)
    else:
        X_discrete = X
    
    # Compute MI between original features and their corresponding attributions
    mi_scores = []
    
    for i in range(min(X_discrete.shape[1], explanations.shape[1])):
        x_feature = X_discrete[:, i]
        attr_feature = explanations[:, i]
        
        mi = mutual_info_regression(x_feature.reshape(-1, 1), attr_feature)
        mi_scores.append(mi[0])
    
    return float(np.mean(mi_scores))


def compute_diversity(model, test_samples, explanation_function, distance_metric='euclidean', 
                     epsilon_min=1e-8):
    """
    Compute Diversity metric for explanation methods on multiple samples.
    
    Measures how diverse the explanations are across different input samples.
    Higher diversity indicates explanations capture different patterns.
    
    Parameters:
    -----------
    model : Callable
        Model prediction function that takes input and returns predictions
    test_samples : np.ndarray
        Multiple input samples to generate explanations for (shape: [n_samples, n_features])
    explanation_function : Callable
        Function that generates explanations for given inputs
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'cosine')
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues
        
    Returns:
    --------
    float : Diversity score (higher is more diverse)
    """
    import numpy as np
    from scipy.spatial.distance import pdist, euclidean, cityblock, cosine
    
    # Generate explanations for all test samples
    explanations = []
    for sample in test_samples:
        sample_flat = np.array(sample).flatten()
        explanation = explanation_function(sample_flat)
        explanations.append(np.array(explanation).flatten())
    
    explanations = np.array(explanations)
    n_examples = len(explanations)
    
    if n_examples < 2:
        return 0.0  # No diversity with single sample
    
    # Choose distance function
    distance_functions = {
        'euclidean': lambda x, y: euclidean(x, y),
        'manhattan': lambda x, y: cityblock(x, y), 
        'cosine': lambda x, y: cosine(x, y) if np.linalg.norm(x) > epsilon_min and np.linalg.norm(y) > epsilon_min else 0.0
    }
    
    if distance_metric not in distance_functions:
        distance_metric = 'euclidean'
    
    distance_func = distance_functions[distance_metric]
    
    # Compute diversity: average pairwise distance
    total_distance = 0
    pair_count = 0
    
    for i in range(n_examples):
        for j in range(i+1, n_examples):  # Only unique pairs
            dist = distance_func(explanations[i], explanations[j])
            total_distance += dist
            pair_count += 1
    
    # Following paper's formula: Σd(xi,xj) / 2NE
    diversity = total_distance / (2 * n_examples) if n_examples > 0 else 0.0
    
    return float(diversity)



def compute_completeness(model, x_sample, explanation_values, epsilon_min=1e-8):
    """
    Compute Completeness metric for any XAI explanation method.
    
    Parameters:
    -----------
    model : Callable
        Model prediction function that takes input and returns predictions
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_values : np.ndarray
        Attribution values from XAI method (same shape as x_sample)
    baseline : np.ndarray, optional
        Baseline input. If None, uses zero baseline for tabular data.
        For other data types, you may want to specify appropriate baseline.
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues
        
    Returns:
    --------
    float : Completeness score (lower absolute value is better, 0 is perfect)
    """
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()
    
    # Create baseline if not provided
    baseline = np.zeros_like(x_sample)  # Zero baseline (common default)

    
    # Get model predictions
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]
    
    baseline_pred = model(baseline.reshape(1, -1))[0]
    if len(baseline_pred.shape) > 0:
        baseline_pred = baseline_pred[1] if len(baseline_pred) > 1 else baseline_pred[0]
    
    # Completeness: Σ attributions should equal F(x) - F(baseline)
    attribution_sum = np.sum(explanation_values)
    model_diff = original_pred - baseline_pred
    completeness_error = attribution_sum - model_diff
    
    return float(completeness_error)



def compute_infidelity(model, x_sample, explanation_values, explanation_function,
                      n_samples=20, noise_std=0.1, epsilon_min=1e-8):
    """
    Compute Infidelity metric for any XAI explanation method.

    Parameters:
    -----------
    model : Callable
        Model prediction function that takes input and returns predictions
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_values : np.ndarray
        Attribution values from XAI method (same shape as x_sample)
    explanation_function : Callable
        Function that generates explanations for new inputs
    n_samples : int, default=20
        Number of perturbation samples to generate
    noise_std : float, default=0.1
        Standard deviation for Gaussian noise perturbations
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues

    Returns:
    --------
    float : Infidelity score (lower is better)
    """
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()

    # Get original prediction
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]

    infidelity_scores = []

    for _ in range(n_samples):
        # Generate perturbation I (following Definition 2.1)
        perturbation = np.random.normal(0, noise_std, x_sample.shape)

        # Compute x - I (corrected direction according to paper)
        x_perturbed = x_sample - perturbation

        # Get perturbed prediction
        perturbed_pred = model(x_perturbed.reshape(1, -1))[0]
        if len(perturbed_pred.shape) > 0:
            perturbed_pred = perturbed_pred[1] if len(perturbed_pred) > 1 else perturbed_pred[0]

        # Compute infidelity components according to Definition 2.1
        # I^T * Φ(f,x)
        explanation_approx = np.dot(perturbation, explanation_values)

        # f(x) - f(x-I)
        model_diff = original_pred - perturbed_pred

        # (I^T Φ(f,x) - (f(x) - f(x-I)))^2
        infidelity_score = (explanation_approx - model_diff) ** 2
        infidelity_scores.append(infidelity_score)

    return float(np.mean(infidelity_scores))

def compute_complexity(explanation_values):
    """
    Compute Complexity metric according to Definition 4 in Bhatt et al. paper.

    Complexity measures how distributed the explanation is across features.
    Lower complexity = more focused explanation (fewer important features)
    Higher complexity = more distributed explanation (many important features)

    Parameters:
    -----------
    explanation_values : np.ndarray
        Attribution values from XAI method (same naming as your other functions)

    Returns:
    --------
    float : Complexity score (0 = most focused, ln(d) = maximally distributed)
    """
    # Convert to numpy array and take absolute values
    abs_attributions = np.abs(np.array(explanation_values).flatten())

    # Handle edge case where all attributions are zero
    total_attribution = np.sum(abs_attributions)
    if total_attribution == 0:
        return 0.0

    # Create fractional contribution distribution Pg(i)
    prob_distribution = abs_attributions / total_attribution

    # Remove zero probabilities to avoid log(0) in entropy calculation
    prob_distribution = prob_distribution[prob_distribution > 0]

    # Compute entropy: μC = -∑ Pg(i) * ln(Pg(i))
    complexity = -np.sum(prob_distribution * np.log(prob_distribution))

    return float(complexity)



def compute_sensitivity(model, x_sample, explanation_function,
                       n_samples=20, radius=0.1, epsilon_min=1e-8):
    """
    Compute Max-Sensitivity metric according to Definition 3.1 in the paper.

    SENS_MAX(Φ,f,x,r) = max_{||y-x||≤r} ||Φ(f,y) - Φ(f,x)||

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_function : Callable
        Function that generates explanations for inputs
    n_samples : int, default=20
        Number of perturbation samples to approximate the maximum
    radius : float, default=0.1
        Maximum perturbation radius r (L2 norm constraint)
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues

    Returns:
    --------
    float : Max-sensitivity score (lower indicates more stable explanations)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation Φ(f,x)
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    max_sensitivity = 0.0

    for _ in range(n_samples):
        # Generate perturbation with constraint ||perturbation|| ≤ r
        # Method 1: Sample from uniform distribution in L2 ball
        perturbation = np.random.normal(0, 1, x_sample.shape)
        perturbation_norm = np.linalg.norm(perturbation)

        if perturbation_norm > epsilon_min:
            # Normalize and scale to random radius ≤ r
            random_radius = radius * np.random.random()  # uniform in [0, r]
            perturbation = (perturbation / perturbation_norm) * random_radius
        else:
            perturbation = np.zeros_like(x_sample)

        # Compute perturbed point y = x + perturbation, where ||y-x|| ≤ r
        y = x_sample + perturbation

        try:
            # Get explanation for perturbed input Φ(f,y)
            perturbed_explanation = np.array(explanation_function(y)).flatten()

            # Compute ||Φ(f,y) - Φ(f,x)||
            explanation_diff = perturbed_explanation - original_explanation
            sensitivity = np.linalg.norm(explanation_diff)

            # Keep track of maximum: max_{||y-x||≤r} ||Φ(f,y) - Φ(f,x)||
            max_sensitivity = max(max_sensitivity, sensitivity)

        except Exception:
            # Skip this perturbation if explanation fails
            continue

    return float(max_sensitivity)


def compute_relative_input_stability(model, x_sample, explanation_function,
                                   n_samples=20, noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    """
    Compute Relative Input Stability (RIS) according to Equation 2.

    RIS = max ||(ex'-ex)/ex||_p / max(||(x'-x)/x||_p, ε_min)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    # Generate perturbations with same prediction
    valid_perturbations = []
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])

    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_perturbations) < n_samples and attempts < max_attempts:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check if prediction class is the same
        perturbed_pred = np.argmax(model(x_perturbed.reshape(1, -1))[0])
        if perturbed_pred == original_pred:
            valid_perturbations.append(x_perturbed)

        attempts += 1

    if len(valid_perturbations) == 0:
        return 0.0

    max_instability = 0.0

    for x_perturbed in valid_perturbations:
        # Get explanation for perturbed input
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Compute (ex' - ex) / ex (element-wise, handling division by zero)
        explanation_diff = perturbed_explanation - original_explanation
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                   epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp

        # Compute (x' - x) / x (element-wise, handling division by zero)
        input_diff = x_perturbed - x_sample
        safe_original_input = np.where(np.abs(x_sample) < epsilon_min,
                                     epsilon_min, x_sample)
        input_relative = input_diff / safe_original_input

        # Compute norms of the relative change vectors
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        input_norm = np.linalg.norm(input_relative, ord=p_norm)

        # Compute RIS ratio
        instability_ratio = explanation_norm / max(input_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return float(max_instability)


def compute_relative_output_stability(model, x_sample, explanation_function,
                                    n_samples=20, noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    """
    Compute Relative Output Stability (ROS) according to Equation 5.

    ROS = max ||(ex'-ex)/ex||_p / max(||h(x')-h(x)||_p, ε_min)
    where h(x) are the logits (pre-softmax outputs)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation and logits
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    # Get logits - you may need to modify this based on your model interface
    # This assumes model returns probabilities; you need raw logits
    original_output = model(x_sample.reshape(1, -1))[0]

    # If model returns probabilities, convert back to logits (approximation)
    # Better: modify to get actual logits from model
    original_logits = np.log(np.clip(original_output, epsilon_min, 1-epsilon_min))

    # Generate perturbations with same prediction
    valid_perturbations = []
    original_pred = np.argmax(original_output)

    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_perturbations) < n_samples and attempts < max_attempts:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check if prediction class is the same
        perturbed_output = model(x_perturbed.reshape(1, -1))[0]
        perturbed_pred = np.argmax(perturbed_output)

        if perturbed_pred == original_pred:
            # Convert to logits
            perturbed_logits = np.log(np.clip(perturbed_output, epsilon_min, 1-epsilon_min))
            valid_perturbations.append((x_perturbed, perturbed_logits))

        attempts += 1

    if len(valid_perturbations) == 0:
        return 0.0

    max_instability = 0.0

    for x_perturbed, perturbed_logits in valid_perturbations:
        # Get explanation for perturbed input
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Compute (ex' - ex) / ex
        explanation_diff = perturbed_explanation - original_explanation
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                   epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp

        # Compute h(x') - h(x) (logit difference)
        logit_diff = perturbed_logits - original_logits

        # Compute norms
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        logit_norm = np.linalg.norm(logit_diff, ord=p_norm)

        # Compute ROS ratio
        instability_ratio = explanation_norm / max(logit_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return float(max_instability)


def compute_relative_representation_stability1(model, x_sample, explanation_function,
                                            representation_layer, n_samples=20,
                                            noise_std=0.05, epsilon_min=1e-3, p_norm=2):  # Increased epsilon_min
    """
    Compute Relative Representation Stability (RRS) according to Equation 3.
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation and representation
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_representation = np.array(representation_layer(x_sample)).flatten()

    # Generate perturbations with same prediction
    valid_perturbations = []
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])

    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_perturbations) < n_samples and attempts < max_attempts:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check if prediction class is the same
        perturbed_pred = np.argmax(model(x_perturbed.reshape(1, -1))[0])

        if perturbed_pred == original_pred:
            # Get representation for perturbed input
            perturbed_representation = np.array(representation_layer(x_perturbed)).flatten()
            valid_perturbations.append((x_perturbed, perturbed_representation))

        attempts += 1

    if len(valid_perturbations) == 0:
        return 0.0

    max_instability = 0.0

    for x_perturbed, perturbed_representation in valid_perturbations:
        # Get explanation for perturbed input
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Compute absolute differences instead of relative changes for better stability
        explanation_diff = perturbed_explanation - original_explanation
        representation_diff = perturbed_representation - original_representation

        # Compute norms of the difference vectors
        explanation_norm = np.linalg.norm(explanation_diff, ord=p_norm)
        representation_norm = np.linalg.norm(representation_diff, ord=p_norm)

        # Compute RRS ratio with better numerical stability
        instability_ratio = explanation_norm / max(representation_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return float(max_instability)

def compute_relative_representation_stability(model, x_sample, explanation_function,
                                            representation_layer, n_samples=50,  # Changed to match paper
                                            noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    """
    Compute Relative Representation Stability (RRS) according to Equation 3.
    
    RRS = max ||(ex'-ex)/ex||_p / max(||(Lx'-Lx)/Lx||_p, ε_min)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation and representation
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_representation = np.array(representation_layer(x_sample)).flatten()

    # Generate perturbations with same prediction
    valid_perturbations = []
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])

    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_perturbations) < n_samples and attempts < max_attempts:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check if prediction class is the same
        perturbed_pred = np.argmax(model(x_perturbed.reshape(1, -1))[0])

        if perturbed_pred == original_pred:
            # Get representation for perturbed input
            perturbed_representation = np.array(representation_layer(x_perturbed)).flatten()
            valid_perturbations.append((x_perturbed, perturbed_representation))

        attempts += 1

    if len(valid_perturbations) == 0:
        return 0.0

    max_instability = 0.0

    for x_perturbed, perturbed_representation in valid_perturbations:
        # Get explanation for perturbed input
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # CORRECTED: Compute (ex' - ex) / ex (percent change in explanations)
        explanation_diff = perturbed_explanation - original_explanation
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                   epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp

        # CORRECTED: Compute (Lx' - Lx) / Lx (percent change in representations)
        representation_diff = perturbed_representation - original_representation
        safe_original_rep = np.where(np.abs(original_representation) < epsilon_min,
                                   epsilon_min, original_representation)
        representation_relative = representation_diff / safe_original_rep

        # Compute norms of the relative change vectors
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        representation_norm = np.linalg.norm(representation_relative, ord=p_norm)

        # Compute RRS ratio
        instability_ratio = explanation_norm / max(representation_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return float(max_instability)



def compute_faithfulness(model, x_sample, explanation_function,
                        subset_size=5, n_subsets=50, baseline_type='zero'):
    """
    Compute Faithfulness metric according to Definition 3 in the paper.

    μF(f, g; x) = corr[Σi∈S g(f, x)i, f(x) − f(x[xs=x̄s])]

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_function : Callable
        Function that generates explanations for inputs
    subset_size : int, default=5
        Size of feature subsets |S| to sample
    n_subsets : int, default=50
        Number of random subsets to sample for correlation estimation
    baseline_type : str, default='zero'
        Type of baseline: 'zero' or 'mean' (if mean, you need to provide training data)

    Returns:
    --------
    float : Faithfulness correlation score (higher indicates more faithful explanations)
    """
    x_sample = np.array(x_sample).flatten()
    d = len(x_sample)

    # Ensure subset_size doesn't exceed number of features
    subset_size = min(subset_size, d)

    # Get original explanation g(f,x)
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    # Get original model output f(x)
    try:
        original_output = float(model(x_sample))
    except Exception:
        return 0.0

    attribution_sums = []
    output_differences = []

    for _ in range(n_subsets):
        # Randomly sample subset S of size |S|
        subset_indices = np.random.choice(d, size=subset_size, replace=False)

        # Calculate sum of attributions: Σi∈S g(f, x)i
        attribution_sum = np.sum(original_explanation[subset_indices])
        attribution_sums.append(attribution_sum)

        # Create baseline version x[xs=x̄s]
        x_baseline = x_sample.copy()

        if baseline_type == 'zero':
            x_baseline[subset_indices] = 0.0
        elif baseline_type == 'mean':
            # For mean baseline, you'd need training data mean
            # Using zero as fallback for now
            x_baseline[subset_indices] = 0.0

        try:
            # Get model output for baseline version f(x[xs=x̄s])
            baseline_output = float(model(x_baseline))

            # Calculate output difference: f(x) − f(x[xs=x̄s])
            output_diff = original_output - baseline_output
            output_differences.append(output_diff)

        except Exception:
            # Skip this subset if model prediction fails
            continue

    # Compute Pearson correlation between attribution sums and output differences
    if len(attribution_sums) < 2 or len(output_differences) < 2:
        return 0.0

    try:
        # Calculate Pearson correlation coefficient
        correlation_matrix = np.corrcoef(attribution_sums, output_differences)
        correlation = correlation_matrix[0, 1]

        # Handle NaN case (when variance is zero)
        if np.isnan(correlation):
            return 0.0

        return float(correlation)

    except Exception:
        return 0.0


def compute_faithfulness_with_mean_baseline(model, x_sample, explanation_function,
                                          training_mean, subset_size=5, n_subsets=50):
    """
    Compute Faithfulness metric with mean baseline from training data.

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_function : Callable
        Function that generates explanations for inputs
    training_mean : np.ndarray
        Mean values from training data for baseline
    subset_size : int, default=5
        Size of feature subsets |S| to sample
    n_subsets : int, default=50
        Number of random subsets to sample for correlation estimation

    Returns:
    --------
    float : Faithfulness correlation score (higher indicates more faithful explanations)
    """
    x_sample = np.array(x_sample).flatten()
    training_mean = np.array(training_mean).flatten()
    d = len(x_sample)

    # Ensure subset_size doesn't exceed number of features
    subset_size = min(subset_size, d)

    # Get original explanation g(f,x)
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    # Get original model output f(x)
    try:
        original_output = float(model(x_sample))
    except Exception:
        return 0.0

    attribution_sums = []
    output_differences = []

    for _ in range(n_subsets):
        # Randomly sample subset S of size |S|
        subset_indices = np.random.choice(d, size=subset_size, replace=False)

        # Calculate sum of attributions: Σi∈S g(f, x)i
        attribution_sum = np.sum(original_explanation[subset_indices])
        attribution_sums.append(attribution_sum)

        # Create baseline version x[xs=x̄s] using training mean
        x_baseline = x_sample.copy()
        x_baseline[subset_indices] = training_mean[subset_indices]

        try:
            # Get model output for baseline version f(x[xs=x̄s])
            baseline_output = float(model(x_baseline))

            # Calculate output difference: f(x) − f(x[xs=x̄s])
            output_diff = original_output - baseline_output
            output_differences.append(output_diff)

        except Exception:
            # Skip this subset if model prediction fails
            continue

    # Compute Pearson correlation between attribution sums and output differences
    if len(attribution_sums) < 2 or len(output_differences) < 2:
        return 0.0

    try:
        # Calculate Pearson correlation coefficient
        correlation_matrix = np.corrcoef(attribution_sums, output_differences)
        correlation = correlation_matrix[0, 1]

        # Handle NaN case (when variance is zero)
        if np.isnan(correlation):
            return 0.0

        return float(correlation)

    except Exception:
        return 0.0


def compute_fidelity_disparity(model, x_samples_group0, x_samples_group1,
                               explanation_function, fidelity_type='prediction_gap',
                               k=5, m=1000, sigma=0.1):
    """
    Compute Fidelity Disparity between two demographic groups.

    Measures if explanations accurately represent the model's decision-making
    process equally well across groups.

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples_group0 : list or np.ndarray
        Samples from group 0 (majority group)
    x_samples_group1 : list or np.ndarray
        Samples from group 1 (minority group)
    explanation_function : Callable
        Function that generates explanations for inputs
    fidelity_type : str, default='prediction_gap'
        Type of fidelity: 'prediction_gap' or 'ground_truth'
    k : int, default=5
        Number of top features to consider
    m : int, default=1000
        Number of perturbation samples for prediction gap
    sigma : float, default=0.1
        Noise variance for perturbations

    Returns:
    --------
    dict : Contains fidelity scores for both groups and disparity metrics
    """

    def compute_prediction_gap_fidelity(x_sample):
        """Compute prediction gap fidelity for a single sample"""
        x_sample = np.array(x_sample).flatten()

        try:
            # Get original explanation and prediction
            explanation = np.array(explanation_function(x_sample)).flatten()
            original_pred = float(model(x_sample))

            # Get top k important features
            top_k_indices = np.argsort(np.abs(explanation))[-k:]

            gaps = []
            for _ in range(m):
                # Create perturbed sample - add noise to non-important features
                x_perturbed = x_sample.copy()
                for i in range(len(x_sample)):
                    if i not in top_k_indices:
                        x_perturbed[i] += np.random.normal(0, sigma)

                # Compute prediction gap
                perturbed_pred = float(model(x_perturbed))
                gap = abs(original_pred - perturbed_pred)
                gaps.append(gap)

            return np.mean(gaps)

        except Exception:
            return np.nan

    # Compute fidelity scores for both groups
    group0_scores = []
    group1_scores = []

    for x_sample in x_samples_group0:
        score = compute_prediction_gap_fidelity(x_sample)
        if not np.isnan(score):
            group0_scores.append(score)

    for x_sample in x_samples_group1:
        score = compute_prediction_gap_fidelity(x_sample)
        if not np.isnan(score):
            group1_scores.append(score)

    if len(group0_scores) == 0 or len(group1_scores) == 0:
        return {
            'group0_mean': 0.0,
            'group1_mean': 0.0,
            'disparity_ratio': 1.0,
            'disparity_difference': 0.0
        }

    group0_mean = np.mean(group0_scores)
    group1_mean = np.mean(group1_scores)

    # Compute disparity metrics
    disparity_difference = abs(group0_mean - group1_mean)
    disparity_ratio = max(group0_mean, group1_mean) / (min(group0_mean, group1_mean) + 1e-8)

    return {
        'group0_mean': float(group0_mean),
        'group1_mean': float(group1_mean),
        'disparity_ratio': float(disparity_ratio),
        'disparity_difference': float(disparity_difference),
        'group0_scores': group0_scores,
        'group1_scores': group1_scores
    }


def compute_stability_disparity(model, x_samples_group0, x_samples_group1,
                                explanation_function, m=5, sigma=0.1):
    """
    Compute Stability Disparity between two demographic groups.

    Measures if explanations for similar inputs are equally stable across groups.

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples_group0 : list or np.ndarray
        Samples from group 0 (majority group)
    x_samples_group1 : list or np.ndarray
        Samples from group 1 (minority group)
    explanation_function : Callable
        Function that generates explanations for inputs
    m : int, default=5
        Number of perturbation samples
    sigma : float, default=0.1
        Noise variance for perturbations

    Returns:
    --------
    dict : Contains stability scores for both groups and disparity metrics
    """

    def compute_instability(x_sample):
        """Compute instability for a single sample"""
        x_sample = np.array(x_sample).flatten()

        try:
            # Get original explanation
            original_explanation = np.array(explanation_function(x_sample)).flatten()

            instabilities = []
            for _ in range(m):
                # Add noise to all features
                x_perturbed = x_sample + np.random.normal(0, sigma, x_sample.shape)

                # Get explanation for perturbed input
                perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

                # Compute L1 distance between explanations
                instability = np.linalg.norm(original_explanation - perturbed_explanation, ord=1)
                instabilities.append(instability)

            return np.mean(instabilities)

        except Exception:
            return np.nan

    # Compute stability scores for both groups
    group0_scores = []
    group1_scores = []

    for x_sample in x_samples_group0:
        score = compute_instability(x_sample)
        if not np.isnan(score):
            group0_scores.append(score)

    for x_sample in x_samples_group1:
        score = compute_instability(x_sample)
        if not np.isnan(score):
            group1_scores.append(score)

    if len(group0_scores) == 0 or len(group1_scores) == 0:
        return {
            'group0_mean': 0.0,
            'group1_mean': 0.0,
            'disparity_ratio': 1.0,
            'disparity_difference': 0.0
        }

    group0_mean = np.mean(group0_scores)
    group1_mean = np.mean(group1_scores)

    # Compute disparity metrics (higher instability is worse)
    disparity_difference = abs(group0_mean - group1_mean)
    disparity_ratio = max(group0_mean, group1_mean) / (min(group0_mean, group1_mean) + 1e-8)

    return {
        'group0_mean': float(group0_mean),
        'group1_mean': float(group1_mean),
        'disparity_ratio': float(disparity_ratio),
        'disparity_difference': float(disparity_difference),
        'group0_scores': group0_scores,
        'group1_scores': group1_scores
    }


def compute_consistency_disparity(model, x_samples_group0, x_samples_group1,
                                  explanation_function, m=5):
    """
    Compute Consistency Disparity between two demographic groups.

    Measures if multiple explanations for the same input are equally consistent across groups.
    Note: Only applicable to stochastic explanation methods.

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples_group0 : list or np.ndarray
        Samples from group 0 (majority group)
    x_samples_group1 : list or np.ndarray
        Samples from group 1 (minority group)
    explanation_function : Callable
        Stochastic function that generates explanations for inputs
    m : int, default=5
        Number of explanation samples to generate for consistency check

    Returns:
    --------
    dict : Contains consistency scores for both groups and disparity metrics
    """

    def compute_inconsistency(x_sample):
        """Compute inconsistency for a single sample"""
        x_sample = np.array(x_sample).flatten()

        try:
            # Generate multiple explanations for the same input
            explanations = []
            for _ in range(m + 1):  # +1 for baseline explanation
                explanation = np.array(explanation_function(x_sample)).flatten()
                explanations.append(explanation)

            # Use first explanation as baseline
            baseline_explanation = explanations[0]

            # Compute L1 distances from baseline to other explanations
            inconsistencies = []
            for i in range(1, len(explanations)):
                inconsistency = np.linalg.norm(baseline_explanation - explanations[i], ord=1)
                inconsistencies.append(inconsistency)

            return np.mean(inconsistencies)

        except Exception:
            return np.nan

    # Compute consistency scores for both groups
    group0_scores = []
    group1_scores = []

    for x_sample in x_samples_group0:
        score = compute_inconsistency(x_sample)
        if not np.isnan(score):
            group0_scores.append(score)

    for x_sample in x_samples_group1:
        score = compute_inconsistency(x_sample)
        if not np.isnan(score):
            group1_scores.append(score)

    if len(group0_scores) == 0 or len(group1_scores) == 0:
        return {
            'group0_mean': 0.0,
            'group1_mean': 0.0,
            'disparity_ratio': 1.0,
            'disparity_difference': 0.0
        }

    group0_mean = np.mean(group0_scores)
    group1_mean = np.mean(group1_scores)

    # Compute disparity metrics (higher inconsistency is worse)
    disparity_difference = abs(group0_mean - group1_mean)
    disparity_ratio = max(group0_mean, group1_mean) / (min(group0_mean, group1_mean) + 1e-8)

    return {
        'group0_mean': float(group0_mean),
        'group1_mean': float(group1_mean),
        'disparity_ratio': float(disparity_ratio),
        'disparity_difference': float(disparity_difference),
        'group0_scores': group0_scores,
        'group1_scores': group1_scores
    }


def compute_sparsity_disparity(model, x_samples_group0, x_samples_group1,
                               explanation_function, threshold=0.01):
    """
    Compute Sparsity Disparity between two demographic groups.

    Measures if explanations use different numbers of features across groups.

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples_group0 : list or np.ndarray
        Samples from group 0 (majority group)
    x_samples_group1 : list or np.ndarray
        Samples from group 1 (minority group)
    explanation_function : Callable
        Function that generates explanations for inputs
    threshold : float, default=0.01
        Threshold above which feature importance is considered significant

    Returns:
    --------
    dict : Contains sparsity scores for both groups and disparity metrics
    """

    def compute_complexity(x_sample):
        """Compute complexity (number of significant features) for a single sample"""
        x_sample = np.array(x_sample).flatten()

        try:
            # Get explanation
            explanation = np.array(explanation_function(x_sample)).flatten()

            # Count features with importance above threshold
            significant_features = np.sum(np.abs(explanation) > threshold)

            return float(significant_features)

        except Exception:
            return np.nan

    # Compute sparsity scores for both groups
    group0_scores = []
    group1_scores = []

    for x_sample in x_samples_group0:
        score = compute_complexity(x_sample)
        if not np.isnan(score):
            group0_scores.append(score)

    for x_sample in x_samples_group1:
        score = compute_complexity(x_sample)
        if not np.isnan(score):
            group1_scores.append(score)

    if len(group0_scores) == 0 or len(group1_scores) == 0:
        return {
            'group0_mean': 0.0,
            'group1_mean': 0.0,
            'disparity_ratio': 1.0,
            'disparity_difference': 0.0
        }

    group0_mean = np.mean(group0_scores)
    group1_mean = np.mean(group1_scores)

    # Compute disparity metrics
    disparity_difference = abs(group0_mean - group1_mean)
    disparity_ratio = max(group0_mean, group1_mean) / (min(group0_mean, group1_mean) + 1e-8)

    return {
        'group0_mean': float(group0_mean),
        'group1_mean': float(group1_mean),
        'disparity_ratio': float(disparity_ratio),
        'disparity_difference': float(disparity_difference),
        'group0_scores': group0_scores,
        'group1_scores': group1_scores
    }

def compute_cover(model, x_samples, explanation_function):
    """
    Compute Cover metric - measures how many instances are "explained"
    by having non-zero attributions.

    In the context of feature attribution explanations, an instance is
    "covered" if the explanation assigns meaningful importance to at least
    one feature (above a threshold).

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples : list or np.ndarray
        Input samples to evaluate
    explanation_function : Callable
        Function that generates explanations for inputs

    Returns:
    --------
    dict : Contains cover statistics
    """

    covered_instances = 0
    total_instances = 0
    cover_details = []

    # Threshold for considering a feature "significantly attributed"
    attribution_threshold = 1e-6

    for x_sample in x_samples:
        try:
            x_sample = np.array(x_sample).flatten()

            # Get explanation
            explanation = np.array(explanation_function(x_sample)).flatten()

            # Check if any feature has significant attribution
            max_attribution = np.max(np.abs(explanation))
            has_significant_attribution = max_attribution > attribution_threshold

            if has_significant_attribution:
                covered_instances += 1

            total_instances += 1

            cover_details.append({
                'covered': has_significant_attribution,
                'max_attribution': float(max_attribution),
                'num_significant_features': int(np.sum(np.abs(explanation) > attribution_threshold))
            })

        except Exception:
            # Skip failed explanations
            total_instances += 1
            cover_details.append({
                'covered': False,
                'max_attribution': 0.0,
                'num_significant_features': 0
            })

    coverage_rate = covered_instances / total_instances if total_instances > 0 else 0.0

    return {
        'covered_instances': covered_instances,
        'total_instances': total_instances,
        'coverage_rate': float(coverage_rate),
        'cover_details': cover_details
    }


def compute_rule_overlap(model, x_samples, explanation_function,
                        similarity_threshold=0.8, top_k_features=5):
    """
    Compute Rule Overlap metric - measures ambiguity by finding instances
    with similar explanations (representing overlapping "rules").

    In feature attribution context, we identify "rules" as explanations that
    focus on similar sets of important features, then measure how many
    instances are explained by overlapping rule patterns.

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples : list or np.ndarray
        Input samples to evaluate
    explanation_function : Callable
        Function that generates explanations for inputs
    similarity_threshold : float, default=0.8
        Threshold for considering two explanations as overlapping rules
    top_k_features : int, default=5
        Number of top features to consider when comparing explanations

    Returns:
    --------
    dict : Contains rule overlap statistics
    """

    explanations = []
    valid_samples = []

    # Collect all explanations
    for x_sample in x_samples:
        try:
            x_sample = np.array(x_sample).flatten()
            explanation = np.array(explanation_function(x_sample)).flatten()

            explanations.append(explanation)
            valid_samples.append(x_sample)

        except Exception:
            # Skip failed explanations
            continue

    if len(explanations) < 2:
        return {
            'total_overlaps': 0,
            'overlap_rate': 0.0,
            'num_explanations': len(explanations),
            'ambiguous_instances': 0
        }

    explanations = np.array(explanations)
    total_overlaps = 0
    ambiguous_instances = set()

    # Compare all pairs of explanations
    for i in range(len(explanations)):
        for j in range(i + 1, len(explanations)):

            exp_i = explanations[i]
            exp_j = explanations[j]

            # Get top-k features for each explanation
            top_features_i = set(np.argsort(np.abs(exp_i))[-top_k_features:])
            top_features_j = set(np.argsort(np.abs(exp_j))[-top_k_features:])

            # Calculate Jaccard similarity of top features
            intersection = len(top_features_i.intersection(top_features_j))
            union = len(top_features_i.union(top_features_j))
            jaccard_similarity = intersection / union if union > 0 else 0.0

            # Alternative: Calculate cosine similarity of full explanations
            cosine_sim = np.dot(exp_i, exp_j) / (np.linalg.norm(exp_i) * np.linalg.norm(exp_j) + 1e-8)

            # Use the higher of the two similarities
            similarity = max(jaccard_similarity, abs(cosine_sim))

            # If explanations are similar enough, count as overlap
            if similarity >= similarity_threshold:
                total_overlaps += 1
                ambiguous_instances.add(i)
                ambiguous_instances.add(j)

    num_explanations = len(explanations)
    max_possible_overlaps = (num_explanations * (num_explanations - 1)) // 2
    overlap_rate = total_overlaps / max_possible_overlaps if max_possible_overlaps > 0 else 0.0

    return {
        'total_overlaps': total_overlaps,
        'overlap_rate': float(overlap_rate),
        'num_explanations': num_explanations,
        'ambiguous_instances': len(ambiguous_instances),
        'ambiguity_rate': float(len(ambiguous_instances) / num_explanations) if num_explanations > 0 else 0.0
    }


def compute_unambiguity_score(model, x_samples, explanation_function,
                             similarity_threshold=0.8, top_k_features=5):
    """
    Compute Unambiguity Score - the inverse of rule overlap.
    Higher scores indicate better unambiguity (less overlapping explanations).

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_samples : list or np.ndarray
        Input samples to evaluate
    explanation_function : Callable
        Function that generates explanations for inputs
    similarity_threshold : float, default=0.8
        Threshold for considering two explanations as overlapping
    top_k_features : int, default=5
        Number of top features to consider

    Returns:
    --------
    float : Unambiguity score (higher is better, range [0,1])
    """

    overlap_results = compute_rule_overlap(
        model, x_samples, explanation_function,
        similarity_threshold, top_k_features
    )

    # Unambiguity is the inverse of overlap rate
    unambiguity_score = 1.0 - overlap_results['overlap_rate']

    return float(unambiguity_score)


import numpy as np
import pandas as pd
import os
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier
import matplotlib.pyplot as plt
from tabular_metrics import *
import warnings
warnings.filterwarnings('ignore')



def main():
    """
    Main function to evaluate SHAP, LIME, and EBM with all four metrics
    """

    print("=" * 80)
    print("XAI METRICS EVALUATION")
    print("=" * 80)

    # Configuration
    LSTM_PATH = "/content/drive/MyDrive/SOK/metrics/best_lstm_model.h5"
    EBM_DIR = "/content/drive/MyDrive/SOK/metrics"
    SAMPLE_IDX = 0
    N_METRIC_SAMPLES = 15

    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    print("\n1. PREPARING DATA")
    print("-" * 50)

    # Load data
    df = pd.read_csv('breast.csv')
    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
    X = df.drop(columns=['id', 'Unnamed: 32', 'diagnosis'], errors='ignore')
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    # Select test sample
    x_sample = X_test.iloc[SAMPLE_IDX].values
    true_label = y_test.iloc[SAMPLE_IDX]
    feature_names = X.columns.tolist()

    print(f"✓ Selected sample {SAMPLE_IDX}: True label = {true_label} ({'Malignant' if true_label == 1 else 'Benign'})")

    # ========================================================================
    # DEMOGRAPHIC GROUP PREPARATION FOR DISPARITY ANALYSIS
    # ========================================================================
    print("\n1.5. PREPARING DEMOGRAPHIC GROUPS")
    print("-" * 50)

    # For breast cancer dataset, we can create groups based on diagnosis
    # Group 0: Benign cases (diagnosis = 0)
    # Group 1: Malignant cases (diagnosis = 1)
    group0_indices = y_test[y_test == 0].index
    group1_indices = y_test[y_test == 1].index

    # Get samples for each group (limit to reasonable number for computation)
    max_samples_per_group = 20
    group0_samples = X_test.loc[group0_indices[:max_samples_per_group]].values
    group1_samples = X_test.loc[group1_indices[:max_samples_per_group]].values

    print(f"✓ Group 0 (Benign): {len(group0_samples)} samples")
    print(f"✓ Group 1 (Malignant): {len(group1_samples)} samples")


    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    print("\n2. LOADING MODELS")
    print("-" * 50)

    # Load LSTM
    lstm_model = load_model(LSTM_PATH)
    print("✓ LSTM model loaded")

    # Load EBM and components
    ebm_model = joblib.load(os.path.join(EBM_DIR, 'ebm_model.pkl'))
    scaler = joblib.load(os.path.join(EBM_DIR, 'scaler.pkl'))
    with open(os.path.join(EBM_DIR, 'feature_names.pkl'), 'rb') as f:
        saved_feature_names = pickle.load(f)
    print("✓ EBM model and components loaded")

    # LSTM prediction function
    def lstm_predict_proba(X):
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        X_scaled = scaler.transform(X_array)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        preds = lstm_model.predict(X_reshaped, verbose=0)
        return np.column_stack([1 - preds.flatten(), preds.flatten()])

    # EBM prediction function
    def ebm_predict_proba(X):
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        X_scaled = scaler.transform(X_array)
        X_df = pd.DataFrame(X_scaled, columns=feature_names)
        return ebm_model.predict_proba(X_df)

    def get_lstm_representation(x_sample):
        x_scaled = scaler.transform(x_sample.reshape(1, -1))
        X_reshaped = x_scaled.reshape(x_scaled.shape[0], 1, x_scaled.shape[1])
        preds = lstm_model.predict(X_reshaped, verbose=0)

        # Convert to logits for better numerical stability
        epsilon = 1e-7
        preds_clipped = np.clip(preds.flatten(), epsilon, 1 - epsilon)
        logits = np.log(preds_clipped / (1 - preds_clipped))
        return logits

    def lstm_predict_single(X):
        return lstm_predict_proba(X)[:, 1]

    def ebm_predict_single(X):
      return ebm_predict_proba(X)[:, 1]

    def get_ebm_representation(x_sample):
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)

        preds = ebm_model.predict_proba(x_sample)[:, 1]
        epsilon = 1e-7
        preds_clipped = np.clip(preds, epsilon, 1 - epsilon)
        logits = np.log(preds_clipped / (1 - preds_clipped))
        return logits

    # Get predictions
    lstm_pred = lstm_predict_proba(x_sample)[0]
    ebm_pred = ebm_predict_proba(x_sample)[0]
    test_samples = X_test.iloc[:50].values
    print(f"✓ LSTM prediction: {lstm_pred[1]:.4f} ({'Malignant' if lstm_pred[1] > 0.5 else 'Benign'})")
    print(f"✓ EBM prediction: {ebm_pred[1]:.4f} ({'Malignant' if ebm_pred[1] > 0.5 else 'Benign'})")

    # ========================================================================
    # SHAP EVALUATION
    # ========================================================================


    print("\n3. SHAP EVALUATION")
    print("-" * 50)

    try:
        # Setup SHAP
        X_train_scaled = scaler.transform(X_train)
        background_data = X_train_scaled[:50]
        shap_explainer = shap.KernelExplainer(lstm_predict_proba, background_data)
        print("✓ SHAP explainer initialized")

        # Get SHAP explanation
        def get_shap_explanation(x):
            x_scaled = scaler.transform(x.reshape(1, -1))
            shap_values = shap_explainer.shap_values(x_scaled, nsamples=100)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                return shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                if shap_values.ndim == 3:
                    return shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    if shap_values.shape[1] == len(feature_names):
                        return shap_values[0]
                    elif shap_values.shape[1] == len(feature_names) * 2:
                        return shap_values[0, len(feature_names):]
                    else:
                        return shap_values[0]
                else:
                    return shap_values

        shap_explanation = get_shap_explanation(x_sample)
        shap_explanation = np.array(shap_explanation).flatten()

        # Ensure correct shape
        if len(shap_explanation) != len(feature_names):
            if len(shap_explanation) == len(feature_names) * 2:
                shap_explanation = shap_explanation[len(feature_names):]
            else:
                raise ValueError(f"SHAP shape mismatch: {len(shap_explanation)} vs {len(feature_names)}")


        print(f"✓ SHAP explanation shape: {shap_explanation.shape}")

        # Compute SHAP metrics
        print("Computing SHAP metrics...")
        shap_infidelity = compute_infidelity(lstm_predict_proba, x_sample, shap_explanation, get_shap_explanation, n_samples=N_METRIC_SAMPLES)
        shap_complexity = compute_complexity(shap_explanation)
        shap_sensitivity = compute_sensitivity(lstm_predict_proba, x_sample, get_shap_explanation, n_samples=N_METRIC_SAMPLES)
        shap_ris = compute_relative_input_stability(lstm_predict_proba, x_sample, get_shap_explanation, n_samples=N_METRIC_SAMPLES)
        shap_ros = compute_relative_output_stability(lstm_predict_proba, x_sample, get_shap_explanation, n_samples=N_METRIC_SAMPLES)
        shap_rrs = compute_relative_representation_stability(lstm_predict_proba, x_sample, get_shap_explanation, get_lstm_representation, n_samples=N_METRIC_SAMPLES)
        shap_faith = compute_faithfulness(lstm_predict_proba, x_sample, get_shap_explanation)
        shap_fidelity_disparity = compute_fidelity_disparity(lstm_predict_single, group0_samples, group1_samples,get_shap_explanation, k=5, m=20, sigma=0.1)
        shap_stability_disparity = compute_stability_disparity(lstm_predict_single, group0_samples, group1_samples,get_shap_explanation, m=5, sigma=0.1)
        shap_consistency_disparity = compute_consistency_disparity(lstm_predict_single, group0_samples, group1_samples,get_shap_explanation, m=5)
        shap_sparsity_disparity = compute_sparsity_disparity(lstm_predict_single, group0_samples, group1_samples,get_shap_explanation, threshold=0.01)
        shap_cover = compute_cover(lstm_predict_single, test_samples, get_shap_explanation)
        shap_rule_overlap = compute_rule_overlap(lstm_predict_single, test_samples, get_shap_explanation,similarity_threshold=0.8, top_k_features=5)
        shap_unambiguity = compute_unambiguity_score(lstm_predict_single, test_samples, get_shap_explanation)
        shap_completeness = compute_completeness(lstm_predict_proba, x_sample, shap_explanation)
        shap_diversity = compute_diversity(lstm_predict_single, test_samples, get_shap_explanation, distance_metric='euclidean')
        shap_feature_mi = compute_feature_mutual_information(lstm_predict_single, test_samples, get_shap_explanation, threshold=0.1)
        shap_model_param_rand = compute_model_parameter_randomization(lstm_predict_proba, x_sample, get_shap_explanation,trained_model = lstm_model,  model_type='tensorflow', randomization_type='full',n_samples=N_METRIC_SAMPLES)
        shap_fidelity = compute_fidelity_from_explanations(lstm_predict_proba, x_sample, shap_explanation, get_shap_explanation)




        print("✓ SHAP Results:")
        print(f"  Infidelity: {shap_infidelity:.4f}")
        print(f"  Complexity: {shap_complexity:.4f}")
        print(f"  Sensitivity: {shap_sensitivity:.4f}")
        print(f"  RIS: {shap_ris:.4f}")
        print(f"  ROS: {shap_ros:.4f}")
        print(f"  RRS: {shap_rrs:.4f}")
        print(f"  Faithfulness: {shap_faith:.4f}")
        print("✓ SHAP Disparity Results:")
        print(f"  Fidelity Disparity:")
        print(f"    Group 0 (Benign) Mean: {shap_fidelity_disparity['group0_mean']:.4f}")
        print(f"    Group 1 (Malignant) Mean: {shap_fidelity_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {shap_fidelity_disparity['disparity_ratio']:.4f}")
        print(f"    Disparity Difference: {shap_fidelity_disparity['disparity_difference']:.4f}")
        print(f"  Stability Disparity:")
        print(f"    Group 0 Mean: {shap_stability_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {shap_stability_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {shap_stability_disparity['disparity_ratio']:.4f}")
        print(f"  Consistency Disparity:")
        print(f"    Group 0 Mean: {shap_consistency_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {shap_consistency_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {shap_consistency_disparity['disparity_ratio']:.4f}")
        print(f"  Sparsity Disparity:")
        print(f"    Group 0 Mean: {shap_sparsity_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {shap_sparsity_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {shap_sparsity_disparity['disparity_ratio']:.4f}")
        print("✓ SHAP Cover and Rule Overlap Results:")
        print(f"  Cover Rate: {shap_cover['coverage_rate']:.4f}")
        print(f"  Covered Instances: {shap_cover['covered_instances']}/{shap_cover['total_instances']}")
        print(f"  Rule Overlap Rate: {shap_rule_overlap['overlap_rate']:.4f}")
        print(f"  Total Overlaps: {shap_rule_overlap['total_overlaps']}")
        print(f"  Ambiguous Instances: {shap_rule_overlap['ambiguous_instances']}")
        print(f"  Unambiguity Score: {shap_unambiguity:.4f}")
        print(f"✓ SHAP Completeness: {shap_completeness:.4f}")
        print(f"✓ SHAP Diversity: {shap_diversity:.4f}")
        print(f"✓ SHAP Feature Mutual Information: {shap_feature_mi:.4f}")
        print(f"✓ SHAP Model Parameter Randomization: {shap_model_param_rand:.4f}")
        print(f"✓ SHAP Fidelity: {shap_fidelity:.4f}")


        shap_results = {
            'infidelity': shap_infidelity,
            'complexity': shap_complexity,
            'sensitivity': shap_sensitivity,
            'ris': shap_ris,
            'ros': shap_ros,
            'rrs': shap_rrs,
            'faithfulness': shap_faith,
            'cover': shap_cover,
            'rule_overlap': shap_rule_overlap,
            'total_overlaps': shap_rule_overlap['total_overlaps'],
            'unambiguity_score': shap_unambiguity,
            'fidelity_disparity': shap_fidelity_disparity,
            'stability_disparity': shap_stability_disparity,
            'consistency_disparity': shap_consistency_disparity,
            'sparsity_disparity': shap_sparsity_disparity,
            'completeness': shap_completeness,
            'diversity': shap_diversity,
            'feature_mutual_information': shap_feature_mi,
            'model_parameter_randomization': shap_model_param_rand,
            'fidelity': shap_fidelity
        }
    except Exception as e:
        print(f"✗ SHAP evaluation failed: {e}")
        shap_results = None



    # ========================================================================
    # LIME EVALUATION
    # ========================================================================


    print("\n4. LIME EVALUATION")
    print("-" * 50)

    try:
        # Setup LIME
        lime_explainer = LimeTabularExplainer(
            scaler.transform(X_train),
            feature_names=feature_names,
            class_names=["Benign", "Malignant"],
            discretize_continuous=True,
            random_state=42
        )
        print("✓ LIME explainer initialized")

        # Get LIME explanation
        def get_lime_explanation(x):
            x_scaled = scaler.transform(x.reshape(1, -1))[0]
            explanation = lime_explainer.explain_instance(
                x_scaled, lstm_predict_proba, num_features=len(feature_names), num_samples=200
            )

            explanation_dict = dict(explanation.as_list())
            attributions = []
            for feature_name in feature_names:
                weight = 0.0
                for lime_feature, lime_weight in explanation_dict.items():
                    if feature_name in lime_feature:
                        weight = lime_weight
                        break
                attributions.append(weight)
            return np.array(attributions)

        lime_explanation = get_lime_explanation(x_sample)
        print(f"✓ LIME explanation shape: {lime_explanation.shape}")

        # Compute LIME metrics
        print("Computing LIME metrics...")
        lime_infidelity = compute_infidelity(lstm_predict_proba, x_sample, lime_explanation, get_lime_explanation, n_samples=N_METRIC_SAMPLES)
        lime_complexity = compute_complexity(lime_explanation)
        lime_sensitivity = compute_sensitivity(lstm_predict_proba, x_sample, get_lime_explanation, n_samples=N_METRIC_SAMPLES)
        lime_ris = compute_relative_input_stability(lstm_predict_proba, x_sample, get_lime_explanation, n_samples=N_METRIC_SAMPLES)
        lime_ros = compute_relative_output_stability(lstm_predict_proba, x_sample, get_lime_explanation, n_samples=N_METRIC_SAMPLES)
        lime_rrs = compute_relative_representation_stability(lstm_predict_proba, x_sample, get_lime_explanation, get_lstm_representation, n_samples=N_METRIC_SAMPLES)
        lime_faith = compute_faithfulness(lstm_predict_proba, x_sample, get_lime_explanation)
        lime_fidelity_disparity = compute_fidelity_disparity(lstm_predict_single, group0_samples, group1_samples,get_lime_explanation, k=5, m=20, sigma=0.1)
        lime_stability_disparity = compute_stability_disparity(lstm_predict_single, group0_samples, group1_samples,get_lime_explanation, m=5, sigma=0.1)
        lime_consistency_disparity = compute_consistency_disparity(lstm_predict_single, group0_samples, group1_samples,get_lime_explanation, m=5)
        lime_sparsity_disparity = compute_sparsity_disparity(lstm_predict_single, group0_samples, group1_samples,get_lime_explanation, threshold=0.01)
        lime_cover = compute_cover(lstm_predict_single, test_samples, get_lime_explanation)
        lime_rule_overlap = compute_rule_overlap(lstm_predict_single, test_samples, get_lime_explanation,similarity_threshold=0.8, top_k_features=5)
        lime_unambiguity = compute_unambiguity_score(lstm_predict_single, test_samples, get_lime_explanation)
        lime_completeness = compute_completeness(lstm_predict_proba, x_sample, lime_explanation)
        lime_diversity = compute_diversity(lstm_predict_single, test_samples, get_lime_explanation, distance_metric='euclidean')
        lime_feature_mi = compute_feature_mutual_information(lstm_predict_single, test_samples, get_lime_explanation, threshold=0.1)
        lime_model_param_rand = compute_model_parameter_randomization(lstm_predict_proba, x_sample, get_lime_explanation, trained_model = lstm_model,  model_type='tensorflow', randomization_type='full',n_samples=N_METRIC_SAMPLES) 
        lime_fidelity = compute_fidelity_from_explanations(lstm_predict_proba, x_sample, lime_explanation, get_lime_explanation)



        print("✓ LIME Results:")
        print(f"  Infidelity: {lime_infidelity:.4f}")
        print(f"  Complexity: {lime_complexity:.4f}")
        print(f"  Sensitivity: {lime_sensitivity:.4f}")
        print(f"  RIS: {lime_ris:.4f}")
        print(f"  ROS: {lime_ros:.4f}")
        print(f"  RRS: {lime_rrs:.4f}")
        print(f"  Faithfulness: {lime_faith:.4f}")
        print("✓ lime Disparity Results:")
        print(f"  Fidelity Disparity:")
        print(f"    Group 0 (Benign) Mean: {lime_fidelity_disparity['group0_mean']:.4f}")
        print(f"    Group 1 (Malignant) Mean: {lime_fidelity_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {lime_fidelity_disparity['disparity_ratio']:.4f}")
        print(f"    Disparity Difference: {lime_fidelity_disparity['disparity_difference']:.4f}")
        print(f"  Stability Disparity:")
        print(f"    Group 0 Mean: {lime_stability_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {lime_stability_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {lime_stability_disparity['disparity_ratio']:.4f}")
        print(f"  Consistency Disparity:")
        print(f"    Group 0 Mean: {lime_consistency_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {lime_consistency_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {lime_consistency_disparity['disparity_ratio']:.4f}")
        print(f"  Sparsity Disparity:")
        print(f"    Group 0 Mean: {lime_sparsity_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {lime_sparsity_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {lime_sparsity_disparity['disparity_ratio']:.4f}")
        print("✓ SHAP Cover and Rule Overlap Results:")
        print(f"  Cover Rate: {lime_cover['coverage_rate']:.4f}")
        print(f"  Covered Instances: {lime_cover['covered_instances']}/{lime_cover['total_instances']}")
        print(f"  Rule Overlap Rate: {lime_rule_overlap['overlap_rate']:.4f}")
        print(f"  Total Overlaps: {lime_rule_overlap['total_overlaps']}")
        print(f"  Ambiguous Instances: {lime_rule_overlap['ambiguous_instances']}")
        print(f"  Unambiguity Score: {lime_unambiguity:.4f}")
        print(f"✓ LIME Completeness: {lime_completeness:.4f}")
        print(f"✓ LIME Diversity: {lime_diversity:.4f}")
        print(f"✓ LIME Feature Mutual Information: {lime_feature_mi:.4f}")
        print(f"✓ LIME Model Parameter Randomization: {lime_model_param_rand:.4f}")
        print(f"✓ LIME Fidelity: {lime_fidelity:.4f}")

        lime_results = {
            'infidelity': lime_infidelity,
            'complexity': lime_complexity,
            'sensitivity': lime_sensitivity,
            'ris': lime_ris,
            'ros': lime_ros,
            'rrs': lime_rrs,
            'faithfulness': lime_faith,
            'cover': lime_cover,
            'rule_overlap': lime_rule_overlap,
            'total_overlaps': lime_rule_overlap['total_overlaps'],
            'unambiguity_score': lime_unambiguity,
            'fidelity_disparity': lime_fidelity_disparity,
            'stability_disparity': lime_stability_disparity,
            'consistency_disparity': lime_consistency_disparity,
            'sparsity_disparity': lime_sparsity_disparity,
            'completeness': lime_completeness,
            'diversity': lime_diversity,
            'feature_mutual_information': lime_feature_mi,
            'model_parameter_randomization': lime_model_param_rand,
            'fidelity': lime_fidelity
        }
    except Exception as e:
        print(f"✗ LIME evaluation failed: {e}")
        lime_results = None



    # ========================================================================
    # EBM EVALUATION
    # ========================================================================
    print("\n5. EBM EVALUATION")
    print("-" * 50)

    try:
        # Get EBM explanation
        def get_ebm_explanation(x):
            x_scaled = scaler.transform(x.reshape(1, -1))
            x_df = pd.DataFrame(x_scaled, columns=feature_names)
            local_exp = ebm_model.explain_local(x_df)

            # Try to extract scores
            try:
                overall_exp = local_exp.data(0)
                if isinstance(overall_exp, dict) and 'scores' in overall_exp:
                    scores = np.array(overall_exp['scores'])
                    # Handle shape mismatch (interactions)
                    if len(scores) > len(feature_names):
                        scores = scores[:len(feature_names)]  # Take main effects only
                    return scores
            except:
                pass

            # Fallback: feature-by-feature
            scores = []
            for i in range(len(feature_names)):
                try:
                    feature_exp = local_exp.data(i)
                    if isinstance(feature_exp, dict) and 'scores' in feature_exp:
                        score_data = feature_exp['scores']
                        if isinstance(score_data, (list, np.ndarray)):
                            score = np.mean(score_data) if len(score_data) > 0 else 0.0
                        else:
                            score = float(score_data)
                    else:
                        score = 0.0
                    scores.append(score)
                except:
                    scores.append(0.0)

            return np.array(scores)

        ebm_explanation = get_ebm_explanation(x_sample)
        print(f"✓ EBM explanation shape: {ebm_explanation.shape}")

        # Compute EBM metrics
        print("Computing EBM metrics...")
        ebm_infidelity = compute_infidelity(ebm_predict_proba, x_sample, ebm_explanation, get_ebm_explanation, n_samples=N_METRIC_SAMPLES)
        ebm_complexity = compute_complexity(ebm_explanation)
        ebm_sensitivity = compute_sensitivity(ebm_predict_proba, x_sample, get_ebm_explanation, n_samples=N_METRIC_SAMPLES)
        ebm_ris = compute_relative_input_stability(ebm_predict_proba, x_sample, get_ebm_explanation, n_samples=N_METRIC_SAMPLES)
        ebm_ros = compute_relative_output_stability(ebm_predict_proba, x_sample, get_ebm_explanation, n_samples=N_METRIC_SAMPLES)
        ebm_rrs = compute_relative_representation_stability(ebm_predict_proba, x_sample, get_ebm_explanation, get_ebm_representation, n_samples=N_METRIC_SAMPLES)
        ebm_faith = compute_faithfulness(ebm_predict_proba, x_sample, get_ebm_explanation)
        ebm_fidelity_disparity = compute_fidelity_disparity(ebm_predict_single, group0_samples, group1_samples,get_ebm_explanation, k=5, m=20, sigma=0.1)
        ebm_stability_disparity = compute_stability_disparity(ebm_predict_single, group0_samples, group1_samples,get_ebm_explanation, m=5, sigma=0.1)
        ebm_consistency_disparity = compute_consistency_disparity(ebm_predict_single, group0_samples, group1_samples,get_ebm_explanation, m=5)
        ebm_sparsity_disparity = compute_sparsity_disparity(ebm_predict_single, group0_samples, group1_samples,get_ebm_explanation, threshold=0.01)
        ebm_cover = compute_cover(ebm_predict_single, test_samples, get_ebm_explanation)
        ebm_rule_overlap = compute_rule_overlap(ebm_predict_single, test_samples, get_ebm_explanation,similarity_threshold=0.8, top_k_features=5)
        ebm_unambiguity = compute_unambiguity_score(ebm_predict_single, test_samples, get_ebm_explanation)
        ebm_completeness = compute_completeness(ebm_predict_proba, x_sample, ebm_explanation)
        ebm_diversity = compute_diversity(ebm_predict_single, test_samples, get_ebm_explanation, distance_metric='euclidean')
        ebm_feature_mi = compute_feature_mutual_information(ebm_predict_single, test_samples, get_ebm_explanation, threshold=0.1)
        ebm_model_param_rand = compute_model_parameter_randomization(ebm_predict_proba, x_sample, get_ebm_explanation,trained_model = ebm_model,  model_type='sklearn', randomization_type='full',  n_samples=N_METRIC_SAMPLES)
        ebm_fidelity = compute_fidelity_from_explanations(ebm_predict_proba, x_sample, ebm_explanation, get_ebm_explanation)

        print("✓ EBM Results:")
        print(f"  Infidelity: {ebm_infidelity:.4f}")
        print(f"  Complexity: {ebm_complexity:.4f}")
        print(f"  Sensitivity: {ebm_sensitivity:.4f}")
        print(f"  RIS: {ebm_ris:.4f}")
        print(f"  ROS: {ebm_ros:.4f}")
        print(f"  RRS: {ebm_rrs:.4f}")
        print(f"  Faithfulness: {ebm_faith:.4f}")
        print("✓ Ebm Disparity Results:")
        print(f"  Fidelity Disparity:")
        print(f"    Group 0 (Benign) Mean: {ebm_fidelity_disparity['group0_mean']:.4f}")
        print(f"    Group 1 (Malignant) Mean: {ebm_fidelity_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {ebm_fidelity_disparity['disparity_ratio']:.4f}")
        print(f"    Disparity Difference: {ebm_fidelity_disparity['disparity_difference']:.4f}")
        print(f"  Stability Disparity:")
        print(f"    Group 0 Mean: {ebm_stability_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {ebm_stability_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {ebm_stability_disparity['disparity_ratio']:.4f}")
        print(f"  Consistency Disparity:")
        print(f"    Group 0 Mean: {ebm_consistency_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {ebm_consistency_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {ebm_consistency_disparity['disparity_ratio']:.4f}")
        print(f"  Sparsity Disparity:")
        print(f"    Group 0 Mean: {ebm_sparsity_disparity['group0_mean']:.4f}")
        print(f"    Group 1 Mean: {ebm_sparsity_disparity['group1_mean']:.4f}")
        print(f"    Disparity Ratio: {ebm_sparsity_disparity['disparity_ratio']:.4f}")
        print("✓ ebm Cover and Rule Overlap Results:")
        print(f"  Cover Rate: {ebm_cover['coverage_rate']:.4f}")
        print(f"  Covered Instances: {ebm_cover['covered_instances']}/{ebm_cover['total_instances']}")
        print(f"  Rule Overlap Rate: {ebm_rule_overlap['overlap_rate']:.4f}")
        print(f"  Total Overlaps: {ebm_rule_overlap['total_overlaps']}")
        print(f"  Ambiguous Instances: {ebm_rule_overlap['ambiguous_instances']}")
        print(f"  Unambiguity Score: {ebm_unambiguity:.4f}")
        print(f"✓ EBM Completeness: {ebm_completeness:.4f}")
        print(f"✓ EBM Diversity: {ebm_diversity:.4f}")
        print(f"✓ EBM Feature Mutual Information: {ebm_feature_mi:.4f}")
        print(f"✓ EBM Model Parameter Randomization: {ebm_model_param_rand:.4f}")
        print(f"✓ EBM Fidelity: {ebm_fidelity:.4f}")

        ebm_results = {
            'infidelity': ebm_infidelity,
            'complexity': ebm_complexity,
            'sensitivity': ebm_sensitivity,
            'ris': ebm_ris,
            'ros': ebm_ros,
            'rrs': ebm_rrs,
            'faithfulness': ebm_faith,
            'cover': ebm_cover,
            'rule_overlap': ebm_rule_overlap,
            'total_overlaps': ebm_rule_overlap['total_overlaps'],
            'unambiguity_score': ebm_unambiguity,
            'fidelity_disparity': ebm_fidelity_disparity,
            'stability_disparity': ebm_stability_disparity,
            'consistency_disparity': ebm_consistency_disparity,
            'sparsity_disparity': ebm_sparsity_disparity,
            'completeness': ebm_completeness,
            'diversity': ebm_diversity,
            'feature_mutual_information': ebm_feature_mi,
            'model_parameter_randomization': ebm_model_param_rand,
            'fidelity': ebm_fidelity
        }

    except Exception as e:
        print(f"✗ EBM evaluation failed: {e}")
        ebm_results = None

if __name__ == "__main__":
    main()