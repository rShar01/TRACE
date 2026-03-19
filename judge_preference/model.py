from dataclasses import dataclass, field
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Any
import os

# from vllm import LLM, SamplingParams
# from together import Together
# import cohere
# from reka.client import Reka
# from google.genai import types
# from google import genai
# from fireworks.client import Fireworks
import argparse
from tenacity import retry, wait_exponential, stop_after_attempt
import torch
import logging
import yaml


@dataclass
class Prompt:
    sys_prompt: Optional[str] = None
    query_prompt: str = ""
    prefill: Optional[str] = None

@dataclass
class ModelConfig:
    model_name: str       # Abreviated API name of model 
    api_key: str = None         # Name of ENV variable in OS

    # For local models
    cache_dir: Optional[str] = None
    device: Optional[str] = None # Will be auto-detected if None
    do_sample: bool = True
    temperature: float = 1.0
    max_length: int = 4096

    # vLLM-specific configurations
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1  # For very large models that need pipeline parallelism
    gpu_memory_utilization: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    
    # Quantization settings
    quantization: Optional[str] = None  # Options: "awq", "gptq", "squeezellm", "fp8", etc.
    quantization_param_path: Optional[str] = None  # Path to quantization config
    load_format: str = "auto"  # "auto", "pt", "safetensors", "npcache", "dummy"
    
    # Additional vLLM performance settings
    max_model_len: Optional[int] = None  # Override model's max length
    block_size: int = 16  # Token block size for PagedAttention
    swap_space: int = 4  # GB of swap space per GPU
    enforce_eager: bool = False  # Disable CUDA graph for debugging
    max_context_len_to_capture: int = 8192  # Max context length for CUDA graph capture
    disable_custom_all_reduce: bool = False  # Disable custom all-reduce kernels

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def get_model_config(file_path: str, binary=False):
    with open(file_path, "rb" if binary else "r") as f:
        data: dict = yaml.safe_load(f)

    config = ModelConfig(**data)
    return config


def get_n_lines(s, n):
    lines = s.split('\n')
    counter = 0
    ret_lines = []
    for line in lines:
        if counter >= n:
            return ret_lines
        
        # has characters
        if any(letter.isalnum() for letter in line):
            counter += 1

        ret_lines.append(line)

    return ret_lines

@dataclass
class Model():
    """
    Handles the actual generation of text from a model.

    Attributes
    ----------
    config : ModelConfig
        Config with all necessary model params
    model : typing.Any
        The actual model itself. You should be able to call self.model.generate() or
        self.model.messages.create() to directly get output from the model.
    tokenizer : typing.Any
        The tokenizer for the model.
        

    Methods
    -------
    load(): 
        Prepares model and tokenizer for generation.
        This is called in get_model() after instantiating the class.
    
    generate(prompts: PromptSet, n_samples: int, stopping_condition: Config) -> List[str]:
        Generates text given the query and system prompts.

    """
    config: ModelConfig 
    model: 'typing.Any' = None
    tokenizer: 'typing.Any' = None
    messages: List[Dict[str, str]] = None

    def load(self, key):
        raise NotImplementedError("Subclass must implement abstract method")

    def generate(self, prompts: Prompt) -> list[str]:
        raise NotImplementedError("Subclass must implement abstract method")


@dataclass
class OpenRouterModel(Model):

    def load(self, key: str) -> None:
        self.model = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=key,
        )

        self.name = self.config.model_name

    def load_without_config(self, model_name, key) -> None:
        self.model = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=key)
        self.name = model_name

    def get_messages(self, prompt: Prompt):
        # if self.messages is None or self.config.include_history is False:
        #     self.messages = [
        #         {"role": "system", "content": sys_prompt},
        #         {"role": "user", "content": query_prompt}
        #     ]
        # else:
        #     self.messages.append({"role": "user", "content": query_prompt})
        if prompt.prefill:
            query = prompt.query_prompt + "\n\n" + prompt.prefill
        else:
            query = prompt.query_prompt
        assert prompt.sys_prompt is not None
        self.messages = [
            {"role": "system", "content": prompt.sys_prompt},
            {"role": "user", "content": query}
        ]

    @retry(wait=wait_exponential(min=1, max=3), stop=stop_after_attempt(3))
    def generate(
        self, 
        prompt: Prompt, 
    ) -> List[str]:
        """ Gets a responses from the OAI model. """

        self.get_messages(prompt)

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            max_tokens=self.config.max_length, 
            temperature=self.config.temperature
        )

        # self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    

@dataclass
class OpenAIModel(Model):

    def load(self, key: str) -> None:
        self.model = OpenAI(
            api_key=key,
        )

        self.name = self.config.model_name

    def load_without_config(self, model_name, key) -> None:
        self.model = OpenAI(api_key=key)
        self.name = model_name

    def get_messages(self, prompt: Prompt):
        # if self.messages is None or self.config.include_history is False:
        #     self.messages = [
        #         {"role": "system", "content": sys_prompt},
        #         {"role": "user", "content": query_prompt}
        #     ]
        # else:
        #     self.messages.append({"role": "user", "content": query_prompt})
        if prompt.prefill:
            query = prompt.query_prompt + "\n\n" + prompt.prefill
        else:
            query = prompt.query_prompt
        self.messages = [
            {"role": "system", "content": prompt.sys_prompt},
            {"role": "user", "content": query}
        ]

    # @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompt: Prompt, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """

        self.get_messages(prompt)

        if self.name in ["o3-mini-2025-01-31"]:
            response = self.model.chat.completions.create(
                model=self.name,
                messages=self.messages,
                n=n_samples,
            )

        else:
            response = self.model.chat.completions.create(
                model=self.name,
                messages=self.messages,
                n=n_samples,
                temperature=self.config.temperature
            )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return [response.choices[i].message.content for i in range(len(response.choices))][0]
    
    def _generate(
        self, 
        sys_prompt: str,
        query_prompt: str,
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query_prompt}
        ]

        response = self.model.chat.completions.create(
            model=self.name,
            messages=messages,
            n=n_samples,
        )

        return [response.choices[i].message.content for i in range(len(response.choices))][0]

@dataclass
class GeminiModel(Model):
    full_model_names = {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-pro": "gemini-2.0-pro",
        "gemini-2.0-pro-exp": "gemini-2.0-pro-exp-02-05",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-03-25"
    }

    def load(self, key: str) -> None:
        """Initialize the Gemini API client with the provided API key."""
        
        self.name = self.full_model_names.get(self.config.model_name, self.config.model_name)
        self.model = genai.Client(api_key=key)

    def get_messages(self, prompt: Prompt):
        """Format the prompt for Gemini API."""
        # Gemini doesn't use a message structure like OpenAI/Claude
        # Instead, we'll prepare the content directly
        if prompt.prefill:
            self.content = prompt.query_prompt + "\n\n" + prompt.prefill
        else:
            self.content = prompt.query_prompt
            
        # Store system prompt separately
        self.sys_prompt = prompt.sys_prompt

    def generate(
        self, 
        prompt: Prompt, 
        n_samples: int = 1,
    ) -> str:
        """Generate text using the Gemini API."""
        
        # Format prompts for Gemini
        self.get_messages(prompt)
        
        # Prepare generation config with minimal settings
        generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
        )
        
        # Only add max_output_tokens if explicitly set and not unlimited
        if hasattr(self.config, 'max_length') and self.config.max_length > 0:
            generation_config.max_output_tokens = self.config.max_length
        
        # Add system instruction if provided
        if self.sys_prompt:
            generation_config.system_instruction = self.sys_prompt
        
        # Generate content
        response = self.model.models.generate_content(
            model=self.name,
            contents=self.content,
            config=generation_config
        )
        # print(response)
        
        # Return just the text as a string, like other models
        return response.text
    

class DeepSeekChatModel(OpenAIModel):
    def load(self, key: str) -> None:
        self.model = OpenAI(
            api_key=key, 
            base_url="https://api.deepseek.com/v1",
        )

        self.name = self.config.model_name
        
        
    def get_messages(self, prompt: Prompt):
        if prompt.prefill:
            query = prompt.query_prompt + "\n\n" + prompt.prefill
        else:
            query = prompt.query_prompt
        assert prompt.sys_prompt is not None
        self.messages = [
            {"role": "system", "content": prompt.sys_prompt},
            {"role": "user", "content": query}
        ]

    @retry(wait=wait_exponential(min=1, max=3), stop=stop_after_attempt(3))
    def generate(
        self, 
        prompts: Prompt, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
    
        self.get_messages(prompts)

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
            stream=False,
        )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        assert len(response.choices) == 1
        return response.choices[0].message.content


class TogetherAIModel(OpenAIModel):
    full_model_names = {
        "qwen-2.5-coder-32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "codellama-34B-instruct": "codellama/CodeLlama-34b-Instruct-hf",
        "gemma-2-27B": "google/gemma-2-27b-it",
    }

    def load(self) -> None:
        self.model = Together()
        self.name = self.full_model_names[self.config.model_name]
    
    def load_without_config(self, model_name) -> None:
        self.model = Together()
        self.name = self.full_model_names[model_name]
        
    def get_messages(self, sys_prompt, query_prompt):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": query_prompt})

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: Prompt, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        query_prompt = query_prompt + "\n\n" + prompts.prefill

        self.get_messages(sys_prompt, query_prompt)

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
        )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        return [response.choices[i].message.content for i in range(len(response.choices))]

    def get_logprobs(
        self, 
        query_prompt: str,
        n_samples: int = 1,
    ) -> List[float]:
        """ Gets a responses from the OAI model. """
        
        self.messages = [
            {"role": "user", "content": query_prompt},
        ]

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
            logprobs=True,
            echo=True,
            max_tokens=0,
        )


        logprobs = response.prompt[0].logprobs.token_logprobs
        tokens = response.prompt[0].logprobs.tokens

        return tokens, logprobs
    
@dataclass
class ClaudeModel(Model):
    full_model_names = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-4.0-sonnet": "claude-sonnet-4-20250514"
    }

    def load(self, key: str) -> None:
        self.name = self.full_model_names[self.config.model_name]
        self.model = Anthropic(
            api_key=key,
        )

    def get_messages(self, prompt: Prompt):
        # if self.messages is None or self.config.include_history is False:
        #     self.messages = [{"role": "user", "content": prompts.query_prompt}]
        # else:
        #     self.messages.append({"role": "user", "content": prompts.query_prompt})
        # return
        query = prompt.query_prompt
        self.messages = [{"role": "user", "content": query}]

    # @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(30))
    def generate(
        self, 
        prompt: Prompt, 
        n_samples: int = 1, 
    ) -> List[str]:
        """ Gets a responses from the Anthropic model. """
        
        
        # populate messages
        self.get_messages(prompt)

        if prompt.prefill:
            self.messages.append({"role": "assistant", "content": prompt.prefill}) 
        if prompt.sys_prompt is None:
            response = self.model.messages.create(
                model=self.name,
                messages=self.messages,
                max_tokens=self.config.max_length, 
                temperature=self.config.temperature
            )
        else:
            response = self.model.messages.create(
                model=self.name,
                system=prompt.sys_prompt,
                messages=self.messages,
                max_tokens=self.config.max_length,
                temperature=self.config.temperature
            )

        response = response.content[0].text
        return response
    
@dataclass
class FireworksModel(Model):
    """
    Model implementation for Fireworks AI API.
    
    Provides an interface for generating text using Fireworks' models through their API.
    """
    
    # Map of simplified model names to their full API identifiers
    # (This can be expanded with more models from Fireworks catalog)
    full_model_names = {
        "fw-qwen2.5-code-32b": "accounts/fireworks/models/qwen-2.5-coder-32b-instruct",
        "llama-3.1-405b-instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct",
    }
    
    def load(self, key: str) -> None:
        """
        Initialize the Fireworks AI client with API key.
        
        Parameters
        ----------
        key : str
            The Fireworks AI API key
        """
        try:
            self.name = self.full_model_names.get(self.config.model_name, self.config.model_name)
            print(f"Loading Fireworks model: {self.name}")
            print(f"Using Fireworks API key: {key}")
            self.model = Fireworks(api_key=key)
        except ImportError:
            raise ImportError(
                "The fireworks-ai package is not installed. "
                "Please install it with `pip install fireworks-ai`"
            )
    
    def get_messages(self, prompt: Prompt):
        """
        Format input prompt into messages for the Fireworks API.
        
        Parameters
        ----------
        prompt : Prompt
            The prompt object containing query_prompt
        """
        query = prompt.query_prompt
        self.messages = [{"role": "user", "content": query}]
    
    def generate(
        self, 
        prompt: Prompt, 
        n_samples: int = 1
    ) -> List[str]:
        """
        Generate text using the Fireworks AI model.
        
        Parameters
        ----------
        prompt : Prompt
            Object containing the query_prompt, sys_prompt, and prefill
        n_samples : int, optional
            Number of completions to generate (default is 1)
            
        Returns
        -------
        str
            The generated response text
        """
        # Set up messages for the API call
        self.get_messages(prompt)
        
        # Add prefill if provided
        if prompt.prefill:
            self.messages.append({"role": "assistant", "content": prompt.prefill})
        
        # Create generation request
        if prompt.sys_prompt is None:
            response = self.model.chat.completions.create(
                model=self.name,
                messages=self.messages,
                max_tokens=self.config.max_length,
                temperature=self.config.temperature,
                n=n_samples
            )
        else:
            # For Fireworks, we add system message at the beginning of the messages list
            system_message = {"role": "system", "content": prompt.sys_prompt}
            messages_with_system = [system_message] + self.messages
            
            response = self.model.chat.completions.create(
                model=self.name,
                messages=messages_with_system,
                max_tokens=self.config.max_length,
                temperature=self.config.temperature,
                n=n_samples
            )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        return response_text

# @dataclass
# class HuggingFaceModel(Model):
#     config: ModelConfig 
#     tokenizer: Optional[AutoTokenizer] = None
#     model: Optional[AutoModelForCausalLM] = None
#     attention: Optional[str] = 'flash_attention_2'
#     _device: Optional[torch.device] = None

#     def __post_init__(self):
#         self._device = torch.device(self.config.device)
#         logging.info(f"Using device: {self._device}")

#     def load(self, key: Optional[str] = None) -> None:
#         """
#         Loads the Hugging Face model and tokenizer based on the model name provided in config.
#         """
#         # Load the Hugging Face tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.model_name,
#             cache_dir=self.config.cache_dir
#         )
#         logging.info(f"Loaded tokenizer for model {self.config.model_name}")
    
#         if 'Qwen' in self.config.model_name:
#             self.stop_criteria = [EosTokenCriteria(self.tokenizer.encode('<|im_end|>', add_special_tokens=False)[0]), EosTokenCriteria(self.tokenizer.eos_token_id)]
#         elif 'gemma' in self.config.model_name:
#             self.stop_criteria = [EosTokenCriteria(self.tokenizer.encode('<end_of_turn>', add_special_tokens=False)[0]), EosTokenCriteria(self.tokenizer.eos_token_id)]
#         elif 'Llama-3.1' in self.config.model_name:
#             self.stop_criteria = [EosTokenCriteria(self.tokenizer.encode('<|eot_id|>', add_special_tokens=False)[0]), EosTokenCriteria(self.tokenizer.eos_token_id)]
#         else:
#             raise ValueError(f"Model {self.config.model_name} not supported. Implement end of turn token stop criteria in model.py")

#         # Load the Hugging Face model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.model_name,
#             cache_dir=self.config.cache_dir,
#             # attn_implementation=self.attention,
#             device_map=self.config.device,
#         )
        
#         # Move model to device and set evaluation mode
#         self.model.to(self._device)
#         self.model.eval()
        
#         # Set default generation config
#         try:
#             generation_config = GenerationConfig(**self.config.to_dict())
#         except:
#             generation_config = GenerationConfig(**self.config)
#         self.model.generation_config = generation_config
        
#         logging.info(f"Loaded model {self.config.model_name} to {self._device}")


#     def get_messages(self, prompt: Prompt):
#         # import pdb; pdb.set_trace()
#         self.messages = [
#             {"role": "system", "content": prompt.sys_prompt},
#             {"role": "user", "content": prompt.query_prompt},
#         ]
#         if 'Llama' in self.config.model_name:
#             self.messages = [
#                 {'role': 'system', 'content': prompt.sys_prompt},
#                 {'role': 'user', 'content': prompt.query_prompt},
#             ]
#         if 'gemma' in self.config.model_name:
#             self.messages = [
#                     {"role": "user", "content": prompt.sys_prompt + '\n\n' + prompt.query_prompt}
#                 ]
          
#     def generate(
#         self,
#         prompts: Prompt,
#         n_samples: int = 1,
#         sampling_config: Optional[Dict[str, Any]] = None,
#         stopping_condition=None,
#     ) -> List[str]:
#         """
#         Generate responses using the model.
        
#         Args:
#             prompts: PromptSet containing system prompt, query prompt, and optional prefill
#             n_samples: Number of samples to generate
#             sampling_config: Optional override for generation configuration
#         """
#         if 'Qwen' in self.config.model_name:
#             prompts.sys_prompt = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant. ' + prompts.sys_prompt

#         self.get_messages(prompts)

#         if prompts.prefill:
#             self.messages.append({"role": "assistant", "content": prompts.prefill})

#         # Prepare inputs
#         if self.config.model_name.startswith("AtlaAI/"):
#             inputs = self.tokenizer(
#                 prompts.query_prompt,
#                 tokenize=False, add_generation_prompt=True
#                 return_tensors="pt",
#                 return_dict=True,
#             ).to(self._device)
#         else:
#             inputs = self.tokenizer.apply_chat_template(
#                 self.messages,
#                 return_tensors="pt",
#                 return_dict=True,
#                 continue_final_message=bool(prompts.prefill),
#                 add_generation_prompt=not bool(prompts.prefill),
#             ).to(self._device)

#         # Set up generation config
#         if sampling_config:
#             generation_config = GenerationConfig(**sampling_config)
#         else:
#             generation_config = self.model.generation_config

#         if generation_config.pad_token_id is None:
#             # Use the tokenizer's pad token id if available
#             if self.tokenizer.pad_token_id is not None:
#                 generation_config.pad_token_id = self.tokenizer.pad_token_id
#             # Otherwise, use eos_token_id as a fallback
#             else:
#                 generation_config.pad_token_id = self.tokenizer.eos_token_id


#         # import pdb; pdb.set_trace()
#         # Generate responses
#         response_list = []
#         with torch.no_grad():
#             for _ in range(n_samples):
#                 outputs = self.model.generate(
#                     inputs.input_ids,
#                     generation_config=generation_config,
#                 )

#                 # Move outputs to CPU if they are on GPU
#                 if self._device.type == 'cuda':
#                     outputs = outputs.cpu()
#                 response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#                 response_list.append(response)

#         self.messages.append({"role": "assistant", "content": response})

#         return response_list

@dataclass
class VLLMModel(Model):
    """
    Enhanced vLLM Model implementation with multi-GPU and quantization support.
    
    This class provides optimized inference for large language models using vLLM
    with support for:
    - Multi-GPU tensor parallelism
    - Pipeline parallelism for very large models
    - Various quantization formats (GPTQ, AWQ, FP8, etc.)
    - Advanced memory management
    """
    config: ModelConfig 
    model: Optional['LLM'] = None
    tokenizer: Optional['AutoTokenizer'] = None
    _stop_tokens: Optional[List[str]] = None

    def __post_init__(self):
        # Set up stop tokens based on model family
        self._setup_stop_tokens()
        # Validate GPU configuration
        self._validate_gpu_config()

    def _validate_gpu_config(self) -> None:
        """Validate GPU configuration and provide warnings/suggestions."""
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. vLLM requires CUDA for GPU inference.")
            return
            
        num_gpus = torch.cuda.device_count()
        tensor_parallel = self.config.tensor_parallel_size
        pipeline_parallel = self.config.pipeline_parallel_size
        
        total_gpus_needed = tensor_parallel * pipeline_parallel
        
        if total_gpus_needed > num_gpus:
            logging.error(
                f"Configuration requires {total_gpus_needed} GPUs "
                f"(tensor_parallel={tensor_parallel} × pipeline_parallel={pipeline_parallel}) "
                f"but only {num_gpus} GPUs are available."
            )
            raise ValueError("Insufficient GPUs for the requested configuration")
        
        if total_gpus_needed < num_gpus:
            logging.info(
                f"Using {total_gpus_needed} out of {num_gpus} available GPUs. "
                f"Consider increasing tensor_parallel_size to utilize all GPUs."
            )
        
        # Memory estimation for 70B model
        if "70b" in self.config.model_name.lower():
            gpu_memory_per_device = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            estimated_memory_needed = 70 / tensor_parallel  # Rough estimate: 1GB per billion parameters
            
            if self.config.quantization:
                if "8" in self.config.quantization.lower() or "w8a8" in self.config.model_name.lower():
                    estimated_memory_needed *= 0.5  # 8-bit quantization roughly halves memory
                elif "4" in self.config.quantization.lower():
                    estimated_memory_needed *= 0.25  # 4-bit quantization
            
            logging.info(
                f"Estimated memory needed per GPU: {estimated_memory_needed:.1f}GB "
                f"(Available: {gpu_memory_per_device:.1f}GB per GPU)"
            )

    def _setup_stop_tokens(self) -> None:
        """Set up stop tokens based on the model name."""
        model_name = self.config.model_name.lower()
        
        if 'qwen' in model_name:
            self._stop_tokens = ['<|im_end|>', '<|endoftext|>']
        elif 'gemma' in model_name:
            self._stop_tokens = ['<end_of_turn>', '<eos>']
        elif 'llama' in model_name or 'selene' in model_name:
            # Selene models are based on Llama 3.1/3.3, so use same stop tokens
            self._stop_tokens = ['<|eot_id|>', '<|end_of_text|>']
        elif 'mistral' in model_name or 'mixtral' in model_name:
            self._stop_tokens = ['</s>']
        elif 'phi' in model_name:
            self._stop_tokens = ['<|end|>', '<|endoftext|>']
        else:
            # Default stop tokens
            self._stop_tokens = ['</s>', '<|endoftext|>', '<|end|>']

    def load(self, key: Optional[str] = None) -> None:
        """
        Load the vLLM model and tokenizer with multi-GPU and quantization support.
        
        Parameters
        ----------
        key : Optional[str]
            API key (not used for local models, kept for interface compatibility)
        """
        logging.info(f"Loading vLLM model: {self.config.model_name}")
        logging.info(f"GPU Configuration: tensor_parallel={self.config.tensor_parallel_size}")

        # Load tokenizer for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Core vLLM configuration (compatible with most versions)
        vllm_config = {
            "model": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "auto",
            "seed": 42,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
        }
        
        # Add quantization configuration
        if self.config.quantization:
            vllm_config["quantization"] = self.config.quantization
            logging.info(f"Using quantization: {self.config.quantization}")
            
            # For GPTQ models like Selene-1-Llama-3.3-70B-GPTQ-W8A8
            if self.config.quantization.lower() == "gptq":
                # vLLM will automatically detect GPTQ configuration from the model
                logging.info("GPTQ quantization detected - vLLM will auto-configure")
                
        # Add optional configurations
        if self.config.cache_dir:
            vllm_config["download_dir"] = self.config.cache_dir
            
        if self.config.max_model_len:
            vllm_config["max_model_len"] = self.config.max_model_len
        
        # Version-dependent parameters - add only if supported
        self._add_version_dependent_params(vllm_config)
        
        try:
            # Load the vLLM model
            self.model = LLM(**vllm_config)
            logging.info(f"Successfully loaded vLLM model {self.config.model_name}")
            
            # Log memory usage
            if torch.cuda.is_available():
                for i in range(self.config.tensor_parallel_size):
                    if i < torch.cuda.device_count():
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                        logging.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, "
                                   f"{memory_reserved:.2f}GB reserved")
                        
        except Exception as e:
            logging.error(f"Failed to load vLLM model: {e}")
            # Provide helpful error messages for common issues
            if "out of memory" in str(e).lower():
                logging.error("GPU out of memory. Try reducing gpu_memory_utilization or "
                            "increasing tensor_parallel_size to distribute across more GPUs")
            elif "quantization" in str(e).lower():
                logging.error("Quantization error. Ensure the model supports the specified "
                            "quantization format or try quantization=None")
            elif "unexpected keyword argument" in str(e).lower():
                logging.error("vLLM version compatibility issue. Using fallback configuration.")
                # Try with minimal config
                self._load_with_minimal_config()
                return
            raise

    def _configure_quantization(self, vllm_config: Dict[str, Any]) -> None:
        """
        Configure quantization settings, handling auto-detection and conflicts.
        
        Parameters
        ----------
        vllm_config : Dict[str, Any]
            The vLLM configuration dictionary to modify
        """
        # First, try to detect if the model has built-in quantization
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Check for built-in quantization in model config
            built_in_quantization = None
            if hasattr(model_config, 'quantization_config') and model_config.quantization_config:
                quant_config = model_config.quantization_config
                if hasattr(quant_config, 'quant_method'):
                    built_in_quantization = quant_config.quant_method
                elif isinstance(quant_config, dict) and 'quant_method' in quant_config:
                    built_in_quantization = quant_config['quant_method']
                    
            # Handle different quantization scenarios
            if built_in_quantization:
                logging.info(f"Model has built-in quantization: {built_in_quantization}")
                
                if self.config.quantization:
                    if self.config.quantization.lower() != built_in_quantization.lower():
                        logging.warning(
                            f"Config specifies quantization '{self.config.quantization}' but model "
                            f"has built-in '{built_in_quantization}'. Using model's built-in quantization."
                        )
                    # Don't override built-in quantization
                    # vLLM will automatically use the model's quantization
                else:
                    logging.info(f"Auto-detected quantization: {built_in_quantization}")
                    
            elif self.config.quantization:
                # No built-in quantization, use user-specified
                vllm_config["quantization"] = self.config.quantization
                logging.info(f"Using user-specified quantization: {self.config.quantization}")
                
            else:
                logging.info("No quantization specified")
                
        except Exception as e:
            logging.warning(f"Could not detect model quantization: {e}")
            # Fallback: only add quantization if explicitly specified
            if self.config.quantization:
                vllm_config["quantization"] = self.config.quantization
                logging.info(f"Using fallback quantization: {self.config.quantization}")

    def _add_version_dependent_params(self, vllm_config: Dict[str, Any]) -> None:
        """Add parameters that are only supported in newer vLLM versions."""
        # Try to add newer parameters, ignore if not supported
        optional_params = [
            ("pipeline_parallel_size", self.config.pipeline_parallel_size),
            ("swap_space", self.config.swap_space),
            ("block_size", self.config.block_size),
            ("enforce_eager", self.config.enforce_eager),
            ("max_context_len_to_capture", self.config.max_context_len_to_capture),
            ("disable_custom_all_reduce", self.config.disable_custom_all_reduce),
            ("load_format", self.config.load_format),
        ]
        
        # Import vLLM to check available parameters
        try:
            from vllm import EngineArgs
            import inspect
            
            # Get the signature of EngineArgs.__init__
            sig = inspect.signature(EngineArgs.__init__)
            available_params = set(sig.parameters.keys())
            
            # Only add parameters that are actually supported
            for param_name, param_value in optional_params:
                if param_name in available_params and param_value is not None:
                    vllm_config[param_name] = param_value
                    
        except Exception as e:
            logging.warning(f"Could not check vLLM version compatibility: {e}")
            # Fallback: add only the most common parameters
            safe_params = ["swap_space", "load_format"]
            for param_name, param_value in optional_params:
                if param_name in safe_params and param_value is not None:
                    vllm_config[param_name] = param_value

    def _load_with_minimal_config(self) -> None:
        """Load model with minimal configuration for maximum compatibility."""
        logging.info("Loading with minimal configuration for compatibility")
        
        minimal_config = {
            "model": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
        }
        
        # Add quantization if specified
        if self.config.quantization:
            minimal_config["quantization"] = self.config.quantization
            
        # Add cache dir if specified
        if self.config.cache_dir:
            minimal_config["download_dir"] = self.config.cache_dir
        
        try:
            self.model = LLM(**minimal_config)
            logging.info("Successfully loaded with minimal configuration")
        except Exception as e:
            logging.error(f"Failed to load even with minimal config: {e}")
            raise

    def _get_sampling_params(self) -> 'SamplingParams':
        """Create vLLM SamplingParams from config."""
        from vllm import SamplingParams
        
        return SamplingParams(
            temperature=self.config.temperature if self.config.do_sample else 0.0,
            max_tokens=self.config.max_length,
            stop=self._stop_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

    def _format_messages(self, prompt: Prompt) -> str:
        """
        Format the prompt using the model's chat template.
        
        Parameters
        ----------
        prompt : Prompt
            The prompt object containing system prompt, query, and optional prefill
            
        Returns
        -------
        str
            The formatted prompt string
        """
        # Build messages list
        messages = []
        
        # Add system message if present
        if prompt.sys_prompt:
            # Handle model-specific system prompt formatting
            if 'qwen' in self.config.model_name.lower():
                # Qwen has a specific system prompt format
                system_content = f'You are Qwen, created by Alibaba Cloud. You are a helpful assistant. {prompt.sys_prompt}'
            elif 'selene' in self.config.model_name.lower():
                # Selene models work best with clear, direct system prompts
                system_content = prompt.sys_prompt
            else:
                system_content = prompt.sys_prompt
            
            messages.append({"role": "system", "content": system_content})
        
        # Add user message
        messages.append({"role": "user", "content": prompt.query_prompt})
        
        # Add assistant prefill if present
        if prompt.prefill:
            messages.append({"role": "assistant", "content": prompt.prefill})
        
        # Handle model-specific message formatting
        if 'gemma' in self.config.model_name.lower() and prompt.sys_prompt:
            # Gemma doesn't support system role, combine with user message
            messages = [
                {"role": "user", "content": f"{prompt.sys_prompt}\n\n{prompt.query_prompt}"}
            ]
            if prompt.prefill:
                messages.append({"role": "assistant", "content": prompt.prefill})
        
        # Use tokenizer's chat template
        try:
            if prompt.prefill:
                # For prefill, we want to continue from the assistant message
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True
                )
            else:
                # Normal generation
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            logging.warning(f"Chat template failed: {e}. Using fallback formatting.")
            # Fallback to simple formatting
            formatted_prompt = self._fallback_format(messages, prompt.prefill is not None)
        
        return formatted_prompt

    def _fallback_format(self, messages: List[Dict[str, str]], has_prefill: bool) -> str:
        """
        Fallback message formatting when chat template fails.
        
        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of message dictionaries
        has_prefill : bool
            Whether there's a prefill message
            
        Returns
        -------
        str
            Formatted prompt string
        """
        formatted = ""
        
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                formatted += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                if has_prefill:
                    formatted += f"Assistant: {msg['content']}"
                else:
                    formatted += f"Assistant: {msg['content']}\n\n"
        
        if not has_prefill:
            formatted += "Assistant: "
            
        return formatted

    def generate(
        self,
        prompt: Prompt,
        n_samples: int = 1,
        sampling_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate responses using vLLM.
        
        Parameters
        ----------
        prompt : Prompt
            Prompt object containing query_prompt, sys_prompt, and optional prefill
        n_samples : int, optional
            Number of samples to generate (default: 1)
        sampling_config : Optional[Dict[str, Any]], optional
            Override sampling parameters
            
        Returns
        -------
        str
            Generated response (returns first response if n_samples > 1)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Format the prompt
        formatted_prompt = self._format_messages(prompt)
        
        # Set up sampling parameters
        if sampling_config:
            from vllm import SamplingParams
            sampling_params = SamplingParams(**sampling_config)
        else:
            sampling_params = self._get_sampling_params()
        
        # Generate responses
        try:
            # vLLM can generate multiple samples in a single call
            sampling_params.n = n_samples
            
            outputs = self.model.generate(
                prompts=[formatted_prompt],
                sampling_params=sampling_params
            )
            
            # Extract generated text
            responses = []
            for output in outputs:
                for generated_output in output.outputs:
                    response_text = generated_output.text
                    responses.append(response_text)
            
            return responses[0] if n_samples == 1 else responses[0]  # Return string for compatibility
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            raise

    def generate_batch(
        self,
        prompts: List[Prompt],
        n_samples: int = 1,
        sampling_config: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """
        Generate responses for multiple prompts in a batch (leveraging vLLM's efficiency).
        
        Parameters
        ----------
        prompts : List[Prompt]
            List of prompt objects
        n_samples : int, optional
            Number of samples to generate per prompt (default: 1)
        sampling_config : Optional[Dict[str, Any]], optional
            Override sampling parameters
            
        Returns
        -------
        List[List[str]]
            List of generated responses for each prompt
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Format all prompts
        formatted_prompts = [self._format_messages(prompt) for prompt in prompts]
        
        # Set up sampling parameters
        if sampling_config:
            from vllm import SamplingParams
            sampling_params = SamplingParams(**sampling_config)
        else:
            sampling_params = self._get_sampling_params()
        
        sampling_params.n = n_samples
        
        # Generate responses for all prompts
        try:
            outputs = self.model.generate(
                prompts=formatted_prompts,
                sampling_params=sampling_params
            )
            
            # Group responses by prompt
            all_responses = []
            for output in outputs:
                prompt_responses = []
                for generated_output in output.outputs:
                    prompt_responses.append(generated_output.text)
                all_responses.append(prompt_responses)
            
            return all_responses
            
        except Exception as e:
            logging.error(f"Batch generation failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns
        -------
        Dict[str, Any]
            Model information including GPU usage, quantization, etc.
        """
        info = {
            "model_name": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "quantization": self.config.quantization,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
        }
        
        if torch.cuda.is_available() and self.model is not None:
            gpu_info = []
            for i in range(min(self.config.tensor_parallel_size, torch.cuda.device_count())):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                gpu_info.append({
                    "gpu_id": i,
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2)
                })
            info["gpu_usage"] = gpu_info
        
        return info

@dataclass
class CohereModel(Model):
    full_model_names = {
        "aya-32B": "c4ai-aya-expanse-32b",
    }
    def load(self, key: str) -> None:
        self.model = cohere.ClientV2(
            api_key=key
        )
        self.name = self.full_model_names[self.config.model_name]

    def get_messages(self, prompts):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                {"role": "system", "content": prompts.sys_prompt},
                {"role": "user", "content": prompts.query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": prompts.query_prompt})

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: Prompt, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        self.get_messages(prompts)
        if prompts.prefill:
            self.messages.append({"role": "assistant", "content": prompts.prefill})

        responses = []

         # If necessary, set stopping condition
        max_tokens = self.get_max_tokens(stopping_condition)

        for i in range(n_samples):
            response = self.model.chat(
               model=self.name,
               messages=self.messages,
               max_tokens=max_tokens,
            )           
            response = response.message.content[0].text

            if stopping_condition is not None and stopping_condition.n_lines is not None:
                response = "\n".join(get_n_lines(response, stopping_condition.n_lines))

            responses.append(response)
            
        
        if prompts.prefill != "":
            self.messages.pop() 
        
        self.messages.append({"role": "assistant", "content": response}) # TODO: fix this, this is terrible

        return responses

@dataclass
class RekaModel(Model):
    full_model_names = {
        "reka-core": "reka-core-20240501",
    }

    def load(self, key: str) -> None:
        self.name = self.full_model_names[self.config.model_name]
        self.model = Reka(
            api_key=key,
        )

    def get_messages(self, prompts):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                  {"role": "user", "content": prompts.sys_prompt + '\n\n' + prompts.query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": prompts.sys_prompt + '\n\n' + prompts.query_prompt})


    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(30))
    def generate(
        self, 
        prompts: Prompt, 
        n_samples: int = 1, 
    ) -> List[str]:
        """ Gets a responses from the Anthropic model. """
        
        
        # populate messages
        self.get_messages(prompts)

        if prompts.prefill != "":
            self.messages.append({"role": "assistant", "content": prompts.prefill}) 

        responses = []

         # If necessary, set stopping condition
        max_tokens = self.get_max_tokens(stopping_condition)
        for i in range(n_samples):
            response = self.model.chat.create(
                model=self.name,
                messages=self.messages,
                # messages=messages,
                max_tokens=max_tokens,
            )

            response = response.responses[0].message.content
            if stopping_condition is not None and stopping_condition.n_lines is not None:
                response = "\n".join(get_n_lines(response, stopping_condition.n_lines))

            responses.append(response)
            
        
        if prompts.prefill != "":
            self.messages.pop() 
        
        self.messages.append({"role": "assistant", "content": response}) # TODO: fix this, this is terrible

        return responses

class GemmaRMModel(Model):
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Gemma-2B-rewardmodel-ft')
        self.device = "mps"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'Ray2333/GRM-Gemma-2B-rewardmodel-ft', torch_dtype=torch.float16, 
            device_map=self.device,
        )

    def generate(
        self,
        prompts: Prompt,
        n_samples: int = 1,
    ) -> str:
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        user_prompt = f"{sys_prompt}\n\n{query_prompt}"

        message_a = [
            {"role": "user", "content": user_prompt}, 
            {"role": "assistant", "content": prompts.answer_a},
        ]

        message_b = [
            {"role": "user", "content": user_prompt}, 
            {"role": "assistant", "content": prompts.answer_b},
        ]

        template_a = self.tokenizer.apply_chat_template(message_a, tokenize=False)
        template_b = self.tokenizer.apply_chat_template(message_b, tokenize=False)

        # Batch tokenize together so padding aligns
        batch = self.tokenizer(
            [template_a, template_b],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )[0]  

        reward_a = outputs[0].item()
        reward_b = outputs[1].item()

        return "<answer>[[A]]</answer>" if reward_a > reward_b else "<answer>[[B]]</answer>"

#--------------------------------------------#


OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "o1-preview",
    "gpt-4-turbo",
    "o3-mini-2025-01-31",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5"
]

DEEPSEEK_CHAT_MODELS = [
    "deepseek-r1",
]

ANTHROPIC_MODEL_NAMES = [
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "claude-3.5-haiku",
    "claude-4.0-sonnet"
]

OPEN_ROUTER_NAMES = [
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "deepseek/deepseek-r1",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4"
]

TOGETHER_MODEL_NAMES = [
    "qwen-2.5-coder-32B",
    "codellama-34B-instruct",
    "gemma-2-27B",
]

HF_MODEL_PREFIX = [
        "Qwen/",
        "google/",
        "meta-llama/",
        "AtlaAI/",
        "prometheus-eval/"
]

COHERE_MODEL_NAMES = [
    "aya-32B"
]

REKA_MODEL_NAMES = [
    "reka-core"
]

GEMINI_MODEL_NAMES = [
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-2.0-pro-exp",
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

FIREWORK_MODEL_NAMES = [
    "fw-qwen2.5-code-32b",
    "llama-3.1-405b-instruct"
]

GEMMA_MODEL_NAMES = [
    "GRM-Gemma-2B"
]

def get_model(config: ModelConfig) -> Model:
    model_name = config.model_name
    api_key = config.api_key

    if model_name in OPENAI_MODEL_NAMES:
        model = OpenAIModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in ANTHROPIC_MODEL_NAMES:
        model = ClaudeModel(config)
        model.load(os.getenv(api_key))
        return model

    elif model_name in GEMINI_MODEL_NAMES:  # Add this block
        model = GeminiModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in FIREWORK_MODEL_NAMES:
        model = FireworksModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in DEEPSEEK_CHAT_MODELS:
        model = DeepSeekChatModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in TOGETHER_MODEL_NAMES:
        model = TogetherAIModel(config)
        model.load()
        return model

    elif model_name in OPEN_ROUTER_NAMES:
        model = OpenRouterModel(config)
        model.load(os.getenv(api_key))
        return model

    elif model_name in COHERE_MODEL_NAMES:
        model = CohereModel(config)
        model.load(os.getenv(api_key))
        return model

    elif model_name in REKA_MODEL_NAMES:
        model = RekaModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in GEMMA_MODEL_NAMES:
        model = GemmaRMModel(config)
        model.load()
        return model
    
    elif model_name.startswith(tuple(HF_MODEL_PREFIX)):
        # model = HuggingFaceModel(config)
        os.environ["VLLM_USE_V1"] = "0"
        model = VLLMModel(config)
        model.load()
        return model
    
    else:
        raise ValueError(f"Model {model_name} not found")
