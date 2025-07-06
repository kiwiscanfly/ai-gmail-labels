"""Ollama client integration with model management."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import structlog
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.config import get_config
from src.core.exceptions import (
    OllamaError, 
    ModelError, 
    TemporaryOllamaError, 
    ModelNotFoundError,
    RetryableError
)
from src.models.ollama import ModelInfo, GenerationResult, ChatMessage, ModelConfig

logger = structlog.get_logger(__name__)


# ModelInfo and GenerationResult are now imported from src.models.ollama


class OllamaModelManager:
    """Manages Ollama models with dynamic switching and health monitoring."""

    def __init__(self):
        """Initialize the Ollama model manager."""
        self.config = get_config()
        self.client = ollama.AsyncClient(host=self.config.ollama.host)
        self.current_model = self.config.ollama.models["primary"]
        self.models_cache: Dict[str, ModelInfo] = {}
        self.model_configs = self.config.ollama.models
        self.keep_alive = self.config.ollama.keep_alive
        self._health_check_task: Optional[asyncio.Task] = None
        self._preload_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the model manager."""
        try:
            # Check Ollama connectivity
            await self._check_connectivity()
            
            # Refresh model cache
            await self.refresh_models()
            
            # Preload primary model
            await self.preload_model(self.current_model)
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            
            logger.info(
                "Ollama model manager initialized",
                host=self.config.ollama.host,
                primary_model=self.current_model,
                available_models=list(self.models_cache.keys())
            )
            
        except Exception as e:
            logger.error("Failed to initialize Ollama model manager", error=str(e))
            raise OllamaError(f"Failed to initialize Ollama: {e}")

    async def shutdown(self) -> None:
        """Shutdown the model manager."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Ollama model manager shutdown")

    async def _check_connectivity(self) -> None:
        """Check Ollama server connectivity."""
        try:
            # Try to list models to verify connectivity
            await self.client.list()
            logger.debug("Ollama connectivity verified")
        except Exception as e:
            logger.error("Ollama connectivity check failed", error=str(e))
            raise OllamaError(f"Cannot connect to Ollama at {self.config.ollama.host}: {e}")

    async def refresh_models(self) -> Dict[str, ModelInfo]:
        """Refresh the cache of available models.
        
        Returns:
            Dictionary of model name to ModelInfo.
        """
        try:
            response = await self.client.list()
            self.models_cache.clear()
            
            # Response.models is a list of Model objects
            for model_obj in response.models:
                model_info = ModelInfo(
                    name=model_obj.model,
                    size=model_obj.size,
                    digest=model_obj.digest,
                    modified_at=model_obj.modified_at.isoformat() if model_obj.modified_at else '',
                    loaded=False  # Will be updated by health checks
                )
                self.models_cache[model_info.name] = model_info
            
            logger.debug("Models cache refreshed", model_count=len(self.models_cache))
            return self.models_cache
            
        except Exception as e:
            logger.error("Failed to refresh models", error=str(e))
            raise OllamaError(f"Failed to refresh models: {e}")

    async def list_models(self) -> List[ModelInfo]:
        """List all available models.
        
        Returns:
            List of ModelInfo objects.
        """
        if not self.models_cache:
            await self.refresh_models()
        return list(self.models_cache.values())

    async def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available.
        
        Args:
            model_name: Name of the model to check.
            
        Returns:
            True if model is available, False otherwise.
        """
        if not self.models_cache:
            await self.refresh_models()
        return model_name in self.models_cache

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RetryableError)
    )
    async def preload_model(self, model_name: str) -> None:
        """Preload a model for faster inference.
        
        Args:
            model_name: Name of the model to preload.
        """
        async with self._preload_lock:
            try:
                if not await self.is_model_available(model_name):
                    raise ModelNotFoundError(f"Model not found: {model_name}")
                
                # Generate empty prompt to load model
                await self.client.generate(
                    model=model_name,
                    prompt="",
                    keep_alive=True if self.keep_alive else False
                )
                
                # Update cache
                if model_name in self.models_cache:
                    self.models_cache[model_name].loaded = True
                    self.models_cache[model_name].last_used = time.time()
                
                logger.info("Model preloaded", model=model_name)
                
            except ollama.ResponseError as e:
                if "model not found" in str(e).lower():
                    raise ModelNotFoundError(f"Model not found: {model_name}")
                else:
                    raise TemporaryOllamaError(f"Failed to preload model: {e}")
            except Exception as e:
                logger.error("Failed to preload model", model=model_name, error=str(e))
                raise TemporaryOllamaError(f"Failed to preload model {model_name}: {e}")

    async def switch_model(self, task_type: str) -> str:
        """Switch to appropriate model for task type.
        
        Args:
            task_type: Type of task (categorization, reasoning, fallback).
            
        Returns:
            Name of the switched model.
        """
        try:
            target_model = self.model_configs.get(task_type, self.model_configs["primary"])
            
            if target_model != self.current_model:
                if not await self.is_model_available(target_model):
                    logger.warning(
                        "Target model not available, using current",
                        target=target_model,
                        current=self.current_model
                    )
                    return self.current_model
                
                # Preload the target model
                await self.preload_model(target_model)
                self.current_model = target_model
                
                logger.info("Model switched", task_type=task_type, model=target_model)
            
            return self.current_model
            
        except Exception as e:
            logger.error("Failed to switch model", task_type=task_type, error=str(e))
            # Return current model as fallback
            return self.current_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RetryableError)
    )
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[bool, str]] = None
    ) -> GenerationResult:
        """Generate text using Ollama model.
        
        Args:
            prompt: The prompt to generate from.
            model: Model to use (defaults to current model).
            system: System message to use.
            template: Template to use for generation.
            context: Context from previous generation.
            stream: Whether to stream the response.
            raw: Whether to use raw mode.
            format: Response format (e.g., "json").
            options: Additional model options.
            keep_alive: Keep model alive after generation.
            
        Returns:
            GenerationResult with response and metadata.
        """
        try:
            model_name = model or self.current_model
            
            # Ensure model is available
            if not await self.is_model_available(model_name):
                # Try fallback model
                fallback_model = self.model_configs.get("fallback")
                if fallback_model and await self.is_model_available(fallback_model):
                    logger.warning(
                        "Using fallback model",
                        requested=model_name,
                        fallback=fallback_model
                    )
                    model_name = fallback_model
                else:
                    raise ModelNotFoundError(f"Model not available: {model_name}")
            
            # Set up generation parameters
            params = {
                "model": model_name,
                "prompt": prompt,
                "stream": stream,
                "raw": raw,
                "keep_alive": keep_alive if keep_alive is not None else self.keep_alive
            }
            
            if system:
                params["system"] = system
            if template:
                params["template"] = template
            if context:
                params["context"] = context
            if format:
                params["format"] = format
            if options:
                params["options"] = options
            
            # Generate response
            start_time = time.time()
            response = await self.client.generate(**params)
            generation_time = time.time() - start_time
            
            # Update model usage
            if model_name in self.models_cache:
                self.models_cache[model_name].last_used = time.time()
            
            # Create result object
            result = GenerationResult(
                content=response['response'],
                model=response['model'],
                prompt_tokens=response.get('prompt_eval_count', 0),
                completion_tokens=response.get('eval_count', 0),
                total_duration_ns=response.get('total_duration', 0),
                load_duration_ns=response.get('load_duration', 0),
                prompt_eval_duration_ns=response.get('prompt_eval_duration', 0),
                eval_duration_ns=response.get('eval_duration', 0)
            )
            
            logger.debug(
                "Text generated",
                model=model_name,
                prompt_length=len(prompt),
                response_length=len(result.content),
                tokens_per_second=result.tokens_per_second,
                generation_time=generation_time
            )
            
            return result
            
        except ollama.ResponseError as e:
            if "model not found" in str(e).lower():
                raise ModelNotFoundError(f"Model not found: {model_name}")
            elif "connection" in str(e).lower():
                raise TemporaryOllamaError(f"Connection error: {e}")
            else:
                raise OllamaError(f"Generation failed: {e}")
        except Exception as e:
            logger.error("Generation failed", model=model_name, error=str(e))
            raise TemporaryOllamaError(f"Generation failed: {e}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[Union[bool, str]] = None
    ) -> GenerationResult:
        """Chat with Ollama model using conversation format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model to use (defaults to current model).
            format: Response format (e.g., "json").
            options: Additional model options.
            stream: Whether to stream the response.
            keep_alive: Keep model alive after generation.
            
        Returns:
            GenerationResult with response and metadata.
        """
        try:
            model_name = model or self.current_model
            
            # Ensure model is available
            if not await self.is_model_available(model_name):
                fallback_model = self.model_configs.get("fallback")
                if fallback_model and await self.is_model_available(fallback_model):
                    logger.warning(
                        "Using fallback model for chat",
                        requested=model_name,
                        fallback=fallback_model
                    )
                    model_name = fallback_model
                else:
                    raise ModelNotFoundError(f"Model not available: {model_name}")
            
            # Set up chat parameters
            params = {
                "model": model_name,
                "messages": messages,
                "stream": stream,
                "keep_alive": keep_alive if keep_alive is not None else self.keep_alive
            }
            
            if format:
                params["format"] = format
            if options:
                params["options"] = options
            
            # Generate chat response
            start_time = time.time()
            response = await self.client.chat(**params)
            generation_time = time.time() - start_time
            
            # Update model usage
            if model_name in self.models_cache:
                self.models_cache[model_name].last_used = time.time()
            
            # Create result object
            result = GenerationResult(
                content=response['message']['content'],
                model=response['model'],
                prompt_tokens=response.get('prompt_eval_count', 0),
                completion_tokens=response.get('eval_count', 0),
                total_duration_ns=response.get('total_duration', 0),
                load_duration_ns=response.get('load_duration', 0),
                prompt_eval_duration_ns=response.get('prompt_eval_duration', 0),
                eval_duration_ns=response.get('eval_duration', 0)
            )
            
            logger.debug(
                "Chat response generated",
                model=model_name,
                message_count=len(messages),
                response_length=len(result.content),
                tokens_per_second=result.tokens_per_second,
                generation_time=generation_time
            )
            
            return result
            
        except ollama.ResponseError as e:
            if "model not found" in str(e).lower():
                raise ModelNotFoundError(f"Model not found: {model_name}")
            elif "connection" in str(e).lower():
                raise TemporaryOllamaError(f"Connection error: {e}")
            else:
                raise OllamaError(f"Chat failed: {e}")
        except Exception as e:
            logger.error("Chat failed", model=model_name, error=str(e))
            raise TemporaryOllamaError(f"Chat failed: {e}")

    async def pull_model(self, model_name: str) -> None:
        """Pull/download a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull.
        """
        try:
            logger.info("Pulling model", model=model_name)
            await self.client.pull(model_name)
            
            # Refresh cache after pulling
            await self.refresh_models()
            
            logger.info("Model pulled successfully", model=model_name)
            
        except Exception as e:
            logger.error("Failed to pull model", model=model_name, error=str(e))
            raise OllamaError(f"Failed to pull model {model_name}: {e}")

    async def delete_model(self, model_name: str) -> None:
        """Delete a model from local storage.
        
        Args:
            model_name: Name of the model to delete.
        """
        try:
            await self.client.delete(model_name)
            
            # Remove from cache
            if model_name in self.models_cache:
                del self.models_cache[model_name]
            
            logger.info("Model deleted", model=model_name)
            
        except Exception as e:
            logger.error("Failed to delete model", model=model_name, error=str(e))
            raise OllamaError(f"Failed to delete model {model_name}: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Ollama and models.
        
        Returns:
            Dictionary with health status information.
        """
        try:
            # Check connectivity
            start_time = time.time()
            models_response = await self.client.list()
            response_time = time.time() - start_time
            
            # Get current model info
            current_model_info = self.models_cache.get(self.current_model)
            
            # Check if configured models are available
            configured_models = {}
            for task_type, model_name in self.model_configs.items():
                model_info = self.models_cache.get(model_name)
                configured_models[task_type] = {
                    "model": model_name,
                    "available": model_name in self.models_cache,
                    "loaded": model_info.loaded if model_info else False
                }
            
            return {
                "status": "healthy",
                "host": self.config.ollama.host,
                "response_time_ms": response_time * 1000,
                "current_model": self.current_model,
                "current_model_loaded": current_model_info.loaded if current_model_info else False,
                "total_models": len(models_response.models),
                "configured_models": configured_models,
                "models_cache_size": len(self.models_cache)
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "host": self.config.ollama.host
            }

    async def _periodic_health_check(self) -> None:
        """Periodic health check task."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                status = await self.get_health_status()
                
                if status["status"] == "unhealthy":
                    logger.warning("Ollama health check failed", status=status)
                else:
                    logger.debug("Ollama health check passed", status=status)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check task error", error=str(e))

    async def get_model_stats(self) -> Dict[str, Any]:
        """Get model usage statistics.
        
        Returns:
            Dictionary with model statistics.
        """
        total_models = len(self.models_cache)
        loaded_models = sum(1 for model in self.models_cache.values() if model.loaded)
        
        # Calculate total size
        total_size = sum(model.size for model in self.models_cache.values())
        
        # Get most recently used models
        recent_models = sorted(
            self.models_cache.values(),
            key=lambda m: m.last_used,
            reverse=True
        )[:5]
        
        return {
            "total_models": total_models,
            "loaded_models": loaded_models,
            "total_size_bytes": total_size,
            "current_model": self.current_model,
            "configured_models": list(self.model_configs.values()),
            "recent_models": [
                {
                    "name": model.name,
                    "last_used": model.last_used,
                    "loaded": model.loaded
                }
                for model in recent_models
            ]
        }


# Global Ollama manager instance
_ollama_manager: Optional[OllamaModelManager] = None


async def get_ollama_manager() -> OllamaModelManager:
    """Get the global Ollama manager instance."""
    global _ollama_manager
    if _ollama_manager is None:
        _ollama_manager = OllamaModelManager()
        await _ollama_manager.initialize()
    return _ollama_manager


async def shutdown_ollama_manager() -> None:
    """Shutdown the global Ollama manager."""
    global _ollama_manager
    if _ollama_manager:
        await _ollama_manager.shutdown()
        _ollama_manager = None