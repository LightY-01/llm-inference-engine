import torch
import torch.nn.functional as F
from model import KVCache

class Request:
    def __init__(self, request_id: int, prompt_tokens: list[int], max_new_tokens: int):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = []
        self.is_finished = False
        self.kv_cache = None  # Will hold list of KVCache objects (one per transformer layer)

    @property
    def total_length(self):
        """Current total length of the request (prompt tokens + generated tokens)"""
        return len(self.prompt_tokens) + len(self.generated_tokens)
    
    def remaining_length(self):
        """Remaining length of the request to be generated"""
        return self.max_new_tokens - len(self.generated_tokens)

    def is_finished(self):
        """Check if the request is finished"""
        return self.remaining_length() == 0

class Scheduler:
    def __init__(self, max_batch_size: int = 4):
        self.waiting_queue = []
        self.active_requests = []
        self.max_batch_size = max_batch_size

    def add_request(self, request: Request):
        """Adds a request to the waiting queue (request pool)."""
        self.waiting_queue.append(request)

    def schedule(self) -> list[Request]:
        """
        Determines which requests should be executed in the next step.
        Promotes requests from waiting_queue to active_requests up to max_batch_size.
        Returns the list of active requests.
        """
        # FCFS policy
        while len(self.active_requests) < self.max_batch_size and self.waiting_queue:
            req = self.waiting_queue.pop(0)
            self.active_requests.append(req)
            
        return self.active_requests

    def remove_finished_requests(self):
        """Removes requests that have finished generation from the active pool."""
        self.active_requests = [req for req in self.active_requests if not req.is_finished]

    @property
    def has_pending_work(self) -> bool:
        """Returns True if there are requests waiting or running."""
        return len(self.waiting_queue) > 0 or len(self.active_requests) > 0

class ExecutionEngine:
    def __init__(self, model, device, pad_token_id=0):
        self.model = model
        self.device = device
        self.pad_token_id = pad_token_id
    
    def execute_batch(self, active_requests: list[Request]):
        """Executes a single forward pass for the given batch of active requests."""
        if not active_requests:
            return
        
        batch_size = len(active_requests)
        num_layers = len(self.model.blocks)
        
        # Prefill: no tokens generated yet. Input: all prompt_tokens.
        # Decode: at least 1 token generated. Input: only the last generated_token.
        input_ids = []
        input_lengths = []
        cached_lengths = []
        
        for req in active_requests:
            if len(req.generated_tokens) == 0:
                # Prefill (prompt processing) phase
                # If prompt is longer than context_length, truncate it to fit
                if len(req.prompt_tokens) > self.model.context_length:
                    req.prompt_tokens = req.prompt_tokens[-self.model.context_length:]
                input_tokens = req.prompt_tokens
                input_len = len(input_tokens)
                cached_len = 0
            else:
                # Decode (token generation) phase
                input_tokens = [req.generated_tokens[-1]]
                input_len = 1
                
                # Truncate request's cache if cached_len + input_len > context_length
                # This ensures we have space for the new input token within the context window
                max_allowed_cache = self.model.context_length - input_len # e.g., 255
                for layer in range(num_layers):
                    req_k = req.kv_cache[layer].cache_k
                    if req_k is not None and req_k.shape[2] > max_allowed_cache:
                        req.kv_cache[layer].cache_k = req.kv_cache[layer].cache_k[:, :, -max_allowed_cache:, :]
                        req.kv_cache[layer].cache_v = req.kv_cache[layer].cache_v[:, :, -max_allowed_cache:, :]
                
                # Read the actual truncated cache length
                cached_len = req.kv_cache[0].cache_k.shape[2] if req.kv_cache[0].cache_k is not None else 0
            
            input_ids.append(input_tokens)
            input_lengths.append(input_len)
            cached_lengths.append(cached_len)
            
        max_input_len = max(input_lengths)
        max_cached_len = max(cached_lengths)
        
        # Pad inputs to max_input_len (Right padding)
        padded_inputs = []
        for tokens in input_ids:
            padding_len = max_input_len - len(tokens)
            padded_inputs.append(tokens + [self.pad_token_id] * padding_len)
        
        input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=self.device)
        
        # Construct the combined attention mask and position IDs for the batch
        t_total = max_cached_len + max_input_len
        attention_mask = torch.zeros((batch_size, t_total), dtype=torch.float32, device=self.device)
        positions = []
        
        for i in range(batch_size):
            cached_len = cached_lengths[i]
            input_len = input_lengths[i]
            # Set valid cached positions to 1
            if cached_len > 0:
                attention_mask[i, :cached_len] = 1.0
            # Set valid input positions to 1
            attention_mask[i, max_cached_len:max_cached_len + input_len] = 1.0
            
            # Position IDs for this request
            req_pos = list(range(cached_len, cached_len + input_len))
            padding_len = max_input_len - len(req_pos)
            padded_pos = req_pos + [0] * padding_len
            positions.append(padded_pos)
            
        positions_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
            
        # Prepare batch KV caches
        # For each layer, we concatenate the padded cache_k and cache_v of the active requests
        batch_kv_caches = [KVCache(context_length=self.model.context_length) for _ in range(num_layers)]
        for cache in batch_kv_caches:
            cache.disable_truncation = True
        
        head_dim = self.model.blocks[0].attention.head_dim
        num_kv_heads = self.model.blocks[0].attention.num_kv_heads
        
        for layer in range(num_layers):
            has_cache = any(req.kv_cache[layer].cache_k is not None for req in active_requests)
            if has_cache:
                k_list = []
                v_list = []
                for req in active_requests:
                    req_k = req.kv_cache[layer].cache_k
                    req_v = req.kv_cache[layer].cache_v
                    
                    if req_k is None:
                        # Pad with zeros if request has no cache yet
                        req_k = torch.zeros((1, num_kv_heads, max_cached_len, head_dim), device=self.device)
                        req_v = torch.zeros((1, num_kv_heads, max_cached_len, head_dim), device=self.device)
                    else:
                        cur_cached_len = req_k.shape[2]
                        if cur_cached_len < max_cached_len:
                            # Pad the cache to max_cached_len
                            padding_len = max_cached_len - cur_cached_len
                            pad_k = torch.zeros((1, num_kv_heads, padding_len, head_dim), device=self.device)
                            pad_v = torch.zeros((1, num_kv_heads, padding_len, head_dim), device=self.device)
                            pad_k = pad_k.to(req_k.dtype)
                            pad_v = pad_v.to(req_v.dtype)
                            req_k = torch.cat([req_k, pad_k], dim=2)
                            req_v = torch.cat([req_v, pad_v], dim=2)
                    k_list.append(req_k)
                    v_list.append(req_v)
                
                batch_kv_caches[layer].cache_k = torch.cat(k_list, dim=0)
                batch_kv_caches[layer].cache_v = torch.cat(v_list, dim=0)
            else:
                batch_kv_caches[layer].cache_k = None
                batch_kv_caches[layer].cache_v = None
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor, kv_caches=batch_kv_caches, attention_mask=attention_mask, positions=positions_tensor)
            
        # Extract updated caches and save them back to individual requests
        for layer in range(num_layers):
            batch_k = batch_kv_caches[layer].cache_k
            batch_v = batch_kv_caches[layer].cache_v
            
            for i, req in enumerate(active_requests):
                cached_len = cached_lengths[i]
                input_len = input_lengths[i]
                
                # Slice out the valid new keys/values
                new_k = batch_k[i, :, max_cached_len:max_cached_len + input_len, :]
                new_v = batch_v[i, :, max_cached_len:max_cached_len + input_len, :]
                
                if cached_len > 0:
                    # Slice out old valid cached keys/values
                    old_k = batch_k[i, :, :cached_len, :]
                    old_v = batch_v[i, :, :cached_len, :]
                    updated_k = torch.cat([old_k, new_k], dim=1)
                    updated_v = torch.cat([old_v, new_v], dim=1)
                else:
                    updated_k = new_k
                    updated_v = new_v
                
                # Store back into the request's cache (unsqueezed shape: 1, num_kv_heads, seq_len, head_dim)
                req.kv_cache[layer].cache_k = updated_k.unsqueeze(0)
                req.kv_cache[layer].cache_v = updated_v.unsqueeze(0)
        
        # Pluck next token, append it, and manage request completion
        for i, req in enumerate(active_requests):
            input_len = input_lengths[i]
            next_token_logits = logits[i, input_len - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            req.generated_tokens.append(next_token_id)
            
            if req.remaining_length() == 0:
                req.is_finished = True

class InferenceServer:
    def __init__(self, model, device, pad_token_id=0, max_batch_size=4):
        self.model = model
        self.device = device
        # Scheduler handles active/waiting pools and scheduling logic
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        # ExecutionEngine is stateless and runs batch iterations
        self.engine = ExecutionEngine(model, device, pad_token_id)

    def add_request(self, request_id: int, prompt_tokens: list[int], max_new_tokens: int) -> Request:
        """Validates sequence bounds, initializes KV Cache buffers, and adds request to the scheduler."""
        total_requested_length = len(prompt_tokens) + max_new_tokens
        if total_requested_length > self.model.context_length:
            raise ValueError(
                f"Request {request_id} rejected: total requested length ({total_requested_length} tokens) "
                f"exceeds model context length ({self.model.context_length} tokens)."
            )
        req = Request(request_id, prompt_tokens, max_new_tokens)
        
        num_layers = len(self.model.blocks) if hasattr(self.model, 'blocks') else 4
        req.kv_cache = [KVCache(context_length=self.model.context_length) for _ in range(num_layers)]
        for cache in req.kv_cache:
            cache.disable_truncation = True
        
        self.scheduler.add_request(req)
        return req

    def step(self):
        """Runs a single iteration by scheduling waiting requests and executing active ones."""
        # Ask scheduler for the active batch of requests to execute
        active_requests = self.scheduler.schedule()
        if not active_requests:
            return
        
        # Delegate execution of the active batch to the stateless ExecutionEngine
        self.engine.execute_batch(active_requests)
        
        # Ask scheduler to remove finished requests from the active pool
        self.scheduler.remove_finished_requests()

    @property
    def request_queue(self):
        """Helper to expose all current requests (active + waiting) in the system."""
        return self.scheduler.active_requests + self.scheduler.waiting_queue