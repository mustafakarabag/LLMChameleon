

class SteeringHook:
    """
    A context manager for applying a steering vector to a model layer.
    
    This class registers a forward hook on a specified layer of a PyTorch model.
    The hook adds a steering vector to the layer's output during the forward pass.
    The hook is automatically removed when the `with` block is exited.
    """
    def __init__(
        self,
        model,
        layer_index,
        token_index,
        steering_vector,
        steering_strength=1.0,
        completely_replace=False,
    ):
        """
        Args:
            model: The Hugging Face transformer model.
            layer_index (int): The index of the layer to hook (e.g., 40).
            token_index (int): The index of the token to apply the steering vector to.
            steering_vector (torch.Tensor): The vector to add to the activations.
            steering_strength (float): A multiplier for the steering vector.
            completely_replace (bool): Whether to completely replace the activations.
        """
        self.model = model
        self.layer_index = layer_index
        self.token_index = token_index
        self.steering_vector = steering_vector
        self.steering_strength = steering_strength
        self.completely_replace = completely_replace
        self.handle = None

    def _get_target_layer(self):
        """
        Retrieves the target layer from the model.
        NOTE: This assumes a standard GPT-like architecture (e.g., Llama, Mistral).
        You may need to adjust the path for different model families.
        """
        # Common path for many decoder-only models
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[self.layer_index]
        # Fallback for other potential architectures
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
             return self.model.transformer.h[self.layer_index]
        else:
            raise AttributeError("Could not find a standard 'layers' attribute on the model. Please adjust the path to the target layer.")

    def _hook_fn(self, module, input, output):
        """The actual hook function that modifies the layer's output."""
        # The output of a transformer layer is often a tuple.
        # The first element is the hidden state tensor.
        hidden_states = output[0]
        assert hidden_states.size(0) == 1, "Expected batch size of 1 for the hook function."
        
        # Prepare the steering vector for addition
        vec_to_add = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        
        if self.completely_replace:
            # If completely_replace is True, replace the hidden states with the steering vector
            vec_to_add = self.steering_strength * vec_to_add.unsqueeze(0).unsqueeze(0)
            output[0][:, self.token_index, :] = vec_to_add
        else:
            # Add the steering vector to activations
            #   Shape of hidden_states: (batch_size, sequence_length, hidden_dim)
            if self.token_index is not None:
                # If a specific token index is provided, apply the steering vector only to that token
                output[0][:, self.token_index, :] = hidden_states[:, self.token_index, :] + (self.steering_strength * vec_to_add)
            else:
                # If no specific token index is provided, apply the steering vector to all tokens
                # This is the default behavior
                output[0][:] = hidden_states + (self.steering_strength * vec_to_add)
        
        return output

    def __enter__(self):
        """Called when entering the 'with' block. Registers the hook."""
        target_layer = self._get_target_layer()
        self.handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Called when exiting the 'with' block. Removes the hook."""
        if self.handle:
            self.handle.remove()
