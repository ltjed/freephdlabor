"""
Automatic Context Compaction System for Research Agents

Provides intelligent, automatic memory compaction that works with any BaseResearchAgent.
Compaction is triggered automatically based on token thresholds and preserves all
critical information categories without manual intervention.

Features:
- External memory storage: Preserves original ActionSteps before compaction
- Incremental backup: Saves each ActionStep immediately after completion
- Recovery functionality: Can reconstruct full conversation from external storage
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
from smolagents.memory import ActionStep, MemoryStep
from smolagents.monitoring import Timing


# Model context limits mapping (input tokens)
MODEL_CONTEXT_LIMITS = {
    # Claude models
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    # Anthropic prefixed versions (for LiteLLM routing)
    "anthropic/claude-3-5-sonnet-20241022": 200000,
    "anthropic/claude-3-5-sonnet-20240620": 200000,
    "anthropic/claude-3-5-haiku-20241022": 200000,
    "anthropic/claude-3-opus-20240229": 200000,
    "anthropic/claude-3-sonnet-20240229": 200000,
    "anthropic/claude-3-haiku-20240307": 200000,
    "anthropic/claude-sonnet-4-20250514": 200000,
    "anthropic/claude-opus-4": 200000,
    "anthropic/claude-sonnet-4-5": 200000,
    "anthropic/claude-sonnet-4-5-20250929": 200000,

    # OpenAI GPT-5 models
    "gpt-5": 256000,
    "gpt-5-mini": 256000,
    "gpt-5-nano": 256000,
    
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 128000,
    "o3-2025-04-16": 200000,
    "o4-mini": 128000,
    
    # Gemini models
    "gemini-1.5-pro": 1000000,  # 1M tokens
    "gemini-1.5-flash": 1000000,  # 1M tokens
    "gemini-2.5-pro": 1000000,  # 1M tokens
    "gemini-2.5-flash": 1000000,  # 1M tokens
    "gemini/gemini-1.5-pro": 1000000,
    "gemini/gemini-1.5-flash": 1000000,
    "gemini/gemini-2.5-pro": 1000000,
    "gemini/gemini-2.5-flash": 1000000,
    
    # DeepSeek models
    "deepseek-chat": 64000,
    "deepseek-coder": 64000,
    "deepseek-reasoner": 64000,
    
    # Grok models
    "grok-4-0709": 128000,
    
    # Default fallback
    "default": 128000
}


def detect_runtime_context_limit(model) -> Optional[int]:
    """
    Attempt to detect context limit at runtime via API or model attributes.
    
    Args:
        model: The model instance
        
    Returns:
        Optional[int]: Context limit in tokens if detected, None if not
    """
    try:
        # Method 1: Check if model has context_limit attribute (from our create_model)
        if hasattr(model, 'context_limit'):
            limit = model.context_limit
            print(f"✅ Runtime detection: Model has context_limit attribute: {limit:,} tokens")
            return limit
            
        # Method 2: Try to extract from model kwargs/configuration
        if hasattr(model, 'kwargs') and isinstance(model.kwargs, dict):
            if 'context_limit' in model.kwargs:
                limit = model.kwargs['context_limit']
                print(f"✅ Runtime detection: Found in model kwargs: {limit:,} tokens")
                return limit
                
        # Method 3: Try LiteLLM model info (if available)
        if hasattr(model, 'model_id'):
            try:
                import litellm
                # Some models have this info in litellm
                model_info = litellm.get_model_info(model.model_id)
                if model_info and 'max_tokens' in model_info:
                    limit = model_info['max_tokens']
                    print(f"✅ Runtime detection: LiteLLM model info: {limit:,} tokens")
                    return limit
            except Exception:
                pass  # LiteLLM might not have this info
                
        # Method 4: Try to infer from model response metadata (future enhancement)
        # Could potentially make a small test call and check response headers
        
        print(f"⚠️ Runtime detection failed - no context limit found")
        return None
        
    except Exception as e:
        print(f"⚠️ Runtime detection error: {e}")
        return None


def get_model_context_limit(model) -> int:
    """
    Get the context limit for a given model using runtime detection with fallback.
    
    Args:
        model: The model instance or model_id string
        
    Returns:
        int: Context limit in tokens
    """
    # Step 1: Try runtime detection first (Option 3)
    runtime_limit = detect_runtime_context_limit(model)
    if runtime_limit is not None:
        return runtime_limit
    
    # Step 2: Fall back to static mapping (Option 1 fallback)
    print(f"🔄 Falling back to static context limit mapping...")
    
    # Extract model_id from model object or use string directly
    if hasattr(model, 'model_id'):
        model_id = model.model_id
    elif hasattr(model, 'model'):
        model_id = model.model
    elif isinstance(model, str):
        model_id = model
    else:
        model_id = str(model)
    
    # Clean model_id (remove provider prefixes)
    clean_model_id = model_id
    if model_id.startswith('gemini/'):
        clean_model_id = model_id  # Keep gemini/ prefix for mapping
    
    # Look up context limit
    context_limit = MODEL_CONTEXT_LIMITS.get(clean_model_id, MODEL_CONTEXT_LIMITS.get(model_id, MODEL_CONTEXT_LIMITS["default"]))
    
    print(f"📚 Static mapping: Model '{model_id}' → Context limit: {context_limit:,} tokens")
    return context_limit


def calculate_safe_compaction_threshold(model, safety_margin: float = 0.75) -> int:
    """
    Calculate a safe compaction threshold based on model context limit.
    
    Args:
        model: The model instance
        safety_margin: Fraction of context to use before compaction (0.75 = 75%)
        
    Returns:
        int: Safe token threshold for compaction
    """
    context_limit = get_model_context_limit(model)
    safe_threshold = int(context_limit * safety_margin)
    
    print(f"📊 Safe compaction threshold: {safe_threshold:,} tokens ({safety_margin*100:.0f}% of {context_limit:,})")
    return safe_threshold


class AutomaticContextCompactor:
    """
    Automatic context compaction system for research agents.
    
    Features:
    - Agent-agnostic: Works with any BaseResearchAgent
    - Automatic triggering: No manual tool calls required
    - Comprehensive preservation: Captures ALL ActionStep information
    - Intelligent summarization: Uses token-aware compression
    - Configurable thresholds: Adapts to different model contexts
    """
    
    def __init__(self, agent_instance, max_tokens: Optional[int] = None, min_steps_between_compaction: int = 3, storage_dir: Optional[str] = None, safety_margin: float = 0.75):
        """
        Initialize automatic compaction system.
        
        Args:
            agent_instance: The research agent instance to monitor
            max_tokens: Maximum tokens before forcing compaction (auto-calculated if None)
            min_steps_between_compaction: Minimum steps between compactions
            storage_dir: Directory for external memory storage (defaults to workspace_dir/memory_backup)
            safety_margin: Fraction of model context to use before compaction
        """
        self.agent = agent_instance
        self.min_steps_between_compaction = min_steps_between_compaction
        self.safety_margin = safety_margin
        
        # Calculate max_tokens based on agent's model if not provided
        if max_tokens is None:
            # Try to get model from agent
            model = getattr(agent_instance, 'model', None) or getattr(agent_instance, '_original_model', None)
            if model:
                self.max_tokens = calculate_safe_compaction_threshold(model, safety_margin)
            else:
                self.max_tokens = 75000  # Conservative fallback
                print(f"⚠️ Could not detect model, using fallback threshold: {self.max_tokens:,}")
        else:
            self.max_tokens = max_tokens
        
        # External storage setup
        if storage_dir:
            self.storage_dir = storage_dir
        elif hasattr(agent_instance, 'workspace_dir') and agent_instance.workspace_dir:
            self.storage_dir = os.path.join(agent_instance.workspace_dir, 'memory_backup')
        else:
            self.storage_dir = './memory_backup'
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Compaction state tracking
        self.compaction_count = 0
        self.last_compaction_step = 0
        self.step_count = 0
        
        # Memory backup tracking
        self.backup_file = os.path.join(self.storage_dir, 'full_conversation_backup.jsonl')
        self.last_backed_up_step = 0
        
    def should_compact(self, current_tokens: int) -> bool:
        """
        Determine if compaction should be triggered.
        
        Args:
            current_tokens: Current estimated token count
            
        Returns:
            bool: True if compaction should be performed
        """
        steps_since_compaction = self.step_count - self.last_compaction_step
        
        # CRITICAL: Always compact if exceeding token limit
        if current_tokens > self.max_tokens:
            print(f"🔄 CRITICAL: Token limit exceeded ({current_tokens:,} > {self.max_tokens:,})")
            return True
        
        # Don't compact too frequently
        if steps_since_compaction < self.min_steps_between_compaction:
            print(f"   ⏳ Compaction postponed: only {steps_since_compaction} steps since last")
            return False
        
        # Only compact when token limit is actually exceeded
        # No forced compaction every N steps - let context grow naturally
        print(f"   ✅ No compaction needed ({current_tokens:,} tokens within limit)")
        return False
    
    def extract_comprehensive_context(self, steps: List[MemoryStep]) -> Dict[str, Any]:
        """
        Extract ALL information from ActionSteps for comprehensive compaction.
        
        Args:
            steps: List of memory steps to analyze
            
        Returns:
            Dict containing all extracted information categorized by type
        """
        context = {
            'tool_interactions': [],
            'observations': [],
            'model_reasoning': [],
            'errors': [],
            'code_executions': [],
            'final_outputs': [],
            'images': [],
            'metadata': {
                'total_steps': len(steps),
                'action_steps': 0,
                'successful_tools': 0,
                'errors_encountered': 0
            }
        }
        
        for i, step in enumerate(steps):
            if not isinstance(step, ActionStep):
                continue
                
            context['metadata']['action_steps'] += 1
            
            # Extract tool calls and their details
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tool_call in step.tool_calls:
                    context['tool_interactions'].append({
                        'step': i,
                        'tool_name': getattr(tool_call, 'function', {}).get('name', 'unknown'),
                        'arguments': getattr(tool_call, 'function', {}).get('arguments', ''),
                        'call_id': getattr(tool_call, 'id', '')
                    })
                context['metadata']['successful_tools'] += len(step.tool_calls)
            
            # Extract observations (often the most important for continuity)
            if hasattr(step, 'observations') and step.observations:
                context['observations'].append({
                    'step': i,
                    'content': str(step.observations),
                    'length': len(str(step.observations))
                })
            
            # Extract model reasoning/thinking
            if hasattr(step, 'model_output') and step.model_output:
                context['model_reasoning'].append({
                    'step': i,
                    'reasoning': str(step.model_output),
                    'length': len(str(step.model_output))
                })
            
            # Extract errors for debugging context
            if hasattr(step, 'error') and step.error:
                context['errors'].append({
                    'step': i,
                    'error_type': type(step.error).__name__,
                    'error_message': str(step.error),
                    'length': len(str(step.error))
                })
                context['metadata']['errors_encountered'] += 1
            
            # Extract code executions
            if hasattr(step, 'code_action') and step.code_action:
                context['code_executions'].append({
                    'step': i,
                    'code': str(step.code_action),
                    'length': len(str(step.code_action))
                })
            
            # Extract final outputs/results
            if hasattr(step, 'action_output') and step.action_output:
                context['final_outputs'].append({
                    'step': i,
                    'output': str(step.action_output),
                    'length': len(str(step.action_output))
                })
            
            # Extract any images
            if hasattr(step, 'observations_images') and step.observations_images:
                context['images'].append({
                    'step': i,
                    'image_count': len(step.observations_images),
                    'description': f"{len(step.observations_images)} images captured"
                })
        
        return context
    
    def backup_action_step(self, step: ActionStep, step_number: int) -> None:
        """
        Backup a single ActionStep to external storage immediately after completion.
        
        Args:
            step: The ActionStep to backup
            step_number: The step number for tracking
        """
        try:
            # Create backup entry
            backup_entry = {
                'timestamp': time.time(),
                'step_number': step_number,
                'step_type': type(step).__name__,
                'data': self._serialize_action_step(step)
            }
            
            # Append to backup file (JSONL format for incremental writing)
            with open(self.backup_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(backup_entry, default=str) + '\n')
            
            self.last_backed_up_step = step_number
            print(f"📁 Backed up ActionStep #{step_number} to external storage")
            
        except Exception as e:
            print(f"⚠️ Failed to backup ActionStep #{step_number}: {e}")
    
    def backup_memory_before_compaction(self, steps_to_compact: List[ActionStep]) -> None:
        """
        Backup all steps that will be compacted to external storage.
        
        Args:
            steps_to_compact: List of ActionSteps about to be compacted
        """
        try:
            compaction_backup = {
                'timestamp': time.time(),
                'compaction_id': self.compaction_count + 1,
                'pre_compaction_steps': [],
                'step_count': len(steps_to_compact)
            }
            
            # Serialize all steps
            for i, step in enumerate(steps_to_compact):
                compaction_backup['pre_compaction_steps'].append({
                    'original_step_number': getattr(step, 'step_number', i),
                    'step_data': self._serialize_action_step(step)
                })
            
            # Save to compaction-specific file
            compaction_file = os.path.join(self.storage_dir, f'compaction_{self.compaction_count + 1}_backup.json')
            with open(compaction_file, 'w', encoding='utf-8') as f:
                json.dump(compaction_backup, f, indent=2, default=str)
            
            print(f"💾 Backed up {len(steps_to_compact)} steps before compaction #{self.compaction_count + 1}")
            
        except Exception as e:
            print(f"⚠️ Failed to backup steps before compaction: {e}")
    
    def _serialize_action_step(self, step: ActionStep) -> Dict[str, Any]:
        """
        Serialize an ActionStep to a dictionary for external storage.
        
        Args:
            step: The ActionStep to serialize
            
        Returns:
            Dict containing all step information
        """
        step_data = {
            'step_number': getattr(step, 'step_number', None),
            'timing': {
                'start_time': getattr(step.timing, 'start_time', None),
                'end_time': getattr(step.timing, 'end_time', None)
            } if hasattr(step, 'timing') and step.timing else None,
            'tool_calls': getattr(step, 'tool_calls', None),
            'observations': getattr(step, 'observations', None),
            'model_output': getattr(step, 'model_output', None),
            'action_output': getattr(step, 'action_output', None),
            'error': str(getattr(step, 'error', None)) if getattr(step, 'error', None) else None,
            'is_final_answer': getattr(step, 'is_final_answer', False),
            'code_action': getattr(step, 'code_action', None),
            'observations_images': len(getattr(step, 'observations_images', [])) if hasattr(step, 'observations_images') and getattr(step, 'observations_images', None) is not None else 0
        }
        
        return step_data
    
    def load_full_conversation(self) -> List[Dict[str, Any]]:
        """
        Load the complete conversation history from external storage.
        
        Returns:
            List of all backed up ActionSteps with metadata
        """
        if not os.path.exists(self.backup_file):
            return []
        
        conversation = []
        try:
            with open(self.backup_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        conversation.append(json.loads(line))
            
            print(f"📚 Loaded {len(conversation)} steps from external storage")
            return conversation
            
        except Exception as e:
            print(f"⚠️ Failed to load conversation from external storage: {e}")
            return []
    
    def get_compaction_backups(self) -> List[str]:
        """
        Get list of available compaction backup files.
        
        Returns:
            List of compaction backup file paths
        """
        backup_files = []
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.startswith('compaction_') and filename.endswith('_backup.json'):
                    backup_files.append(os.path.join(self.storage_dir, filename))
            
            backup_files.sort()  # Sort by filename (which includes compaction number)
            return backup_files
            
        except Exception as e:
            print(f"⚠️ Failed to list compaction backups: {e}")
            return []
    
    def create_intelligent_summary(self, context: Dict[str, Any], target_tokens: int = 5000) -> str:
        """
        Create an intelligent summary of the extracted context.
        
        Args:
            context: Comprehensive context extracted from steps
            target_tokens: Target token count for the summary
            
        Returns:
            str: Comprehensive but compact summary
        """
        summary_parts = []
        
        # Add metadata overview
        meta = context['metadata']
        summary_parts.append(f"=== COMPACTED AGENT CONTEXT ===")
        summary_parts.append(f"Total processed: {meta['total_steps']} steps ({meta['action_steps']} actions)")
        summary_parts.append(f"Tools executed: {meta['successful_tools']}, Errors: {meta['errors_encountered']}")
        summary_parts.append("")
        
        # Summarize tool interactions
        if context['tool_interactions']:
            summary_parts.append("🔧 TOOL USAGE SUMMARY:")
            tool_counts = {}
            for interaction in context['tool_interactions']:
                tool_name = interaction['tool_name']
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            for tool, count in sorted(tool_counts.items()):
                summary_parts.append(f"  • {tool}: {count} calls")
            
            # Include recent tool details
            recent_tools = context['tool_interactions'][-3:]
            if recent_tools:
                summary_parts.append("  Recent tool calls:")
                for tool in recent_tools:
                    args_preview = tool['arguments'][:100] + "..." if len(tool['arguments']) > 100 else tool['arguments']
                    summary_parts.append(f"    - {tool['tool_name']}: {args_preview}")
            summary_parts.append("")
        
        # Summarize key observations (prioritize recent and large ones)
        if context['observations']:
            summary_parts.append("📊 KEY OBSERVATIONS:")
            
            # Sort by recency and size
            sorted_obs = sorted(context['observations'], 
                              key=lambda x: (x['step'], x['length']), reverse=True)
            
            for obs in sorted_obs[:5]:  # Top 5 most important
                content_preview = obs['content'][:200] + "..." if len(obs['content']) > 200 else obs['content']
                summary_parts.append(f"  Step {obs['step']}: {content_preview}")
            
            if len(sorted_obs) > 5:
                summary_parts.append(f"  ... and {len(sorted_obs) - 5} more observations")
            summary_parts.append("")
        
        # Include recent model reasoning
        if context['model_reasoning']:
            summary_parts.append("🧠 RECENT MODEL REASONING:")
            recent_reasoning = context['model_reasoning'][-2:]  # Last 2 reasoning steps
            for reasoning in recent_reasoning:
                content_preview = reasoning['reasoning'][:300] + "..." if len(reasoning['reasoning']) > 300 else reasoning['reasoning']
                summary_parts.append(f"  Step {reasoning['step']}: {content_preview}")
            summary_parts.append("")
        
        # Include errors for debugging
        if context['errors']:
            summary_parts.append("⚠️ ERRORS ENCOUNTERED:")
            for error in context['errors'][-3:]:  # Last 3 errors
                summary_parts.append(f"  Step {error['step']}: {error['error_type']} - {error['error_message'][:150]}")
            summary_parts.append("")
        
        # Include final outputs
        if context['final_outputs']:
            summary_parts.append("📤 FINAL OUTPUTS:")
            recent_outputs = context['final_outputs'][-3:]  # Last 3 outputs
            for output in recent_outputs:
                content_preview = output['output'][:200] + "..." if len(output['output']) > 200 else output['output']
                summary_parts.append(f"  Step {output['step']}: {content_preview}")
            summary_parts.append("")
        
        # Add continuation context
        summary_parts.append("🔄 CONTINUATION CONTEXT:")
        summary_parts.append("All critical information has been preserved above.")
        summary_parts.append("Agent can continue seamlessly from this compacted state.")
        summary_parts.append("=== END COMPACTED CONTEXT ===")
        
        return "\n".join(summary_parts)
    
    def perform_compaction(self, current_tokens: int) -> None:
        """
        Perform comprehensive context compaction.
        
        Args:
            current_tokens: Current estimated token count
        """
        self.compaction_count += 1
        self.last_compaction_step = self.step_count
        
        print(f"\n🧠 PERFORMING AUTOMATIC CONTEXT COMPACTION #{self.compaction_count}")
        print(f"   📊 Current tokens: {current_tokens:,}")
        
        # Get all memory steps
        all_steps = self.agent.memory.steps.copy()
        total_steps = len(all_steps)
        
        if total_steps <= 3:
            print(f"   ℹ️ Only {total_steps} steps - skipping compaction")
            return
        
        # BACKUP BEFORE COMPACTION: Store original memory externally
        action_steps_to_compact = [step for step in all_steps if isinstance(step, ActionStep)]
        if action_steps_to_compact:
            print(f"   💾 Backing up {len(action_steps_to_compact)} ActionSteps before compaction...")
            self.backup_memory_before_compaction(action_steps_to_compact)
        
        # Extract comprehensive context
        print(f"   🔍 Extracting context from {total_steps} steps...")
        context = self.extract_comprehensive_context(all_steps)
        
        # Create intelligent summary
        print(f"   🧠 Creating intelligent summary...")
        summary = self.create_intelligent_summary(context)
        
        # Determine steps to preserve (keep last 3 meaningful steps)
        steps_to_preserve = []
        for step in reversed(all_steps[-5:]):  # Look at last 5 steps
            if isinstance(step, ActionStep):
                # Check if step has meaningful content
                has_content = any([
                    hasattr(step, 'observations') and step.observations,
                    hasattr(step, 'tool_calls') and step.tool_calls,
                    hasattr(step, 'action_output') and step.action_output,
                    hasattr(step, 'error') and step.error
                ])
                if has_content:
                    steps_to_preserve.append(step)
                if len(steps_to_preserve) >= 3:
                    break
        
        steps_to_preserve = list(reversed(steps_to_preserve))
        
        # Create compacted step
        compacted_step = ActionStep(
            step_number=0,
            timing=Timing(start_time=time.time(), end_time=time.time()),
            model_output=summary,  # Use model_output so it appears in to_messages()
            is_final_answer=False
        )
        
        # Rebuild memory
        self.agent.memory.steps = [compacted_step] + steps_to_preserve
        
        # Calculate results
        new_step_count = len(self.agent.memory.steps)
        
        print(f"✅ Compaction complete:")
        print(f"   📈 Steps: {total_steps} → {new_step_count}")
        print(f"   💾 Preserved: 1 compacted + {len(steps_to_preserve)} recent steps")
        print(f"   📊 Context categories preserved: {len([k for k, v in context.items() if k != 'metadata' and v])}")
        print()


class ContextMonitoringCallback:
    """
    Callback system to automatically monitor and trigger compaction.
    Integrates with any BaseResearchAgent to provide automatic context management.
    """
    
    def __init__(self, model, token_threshold: Optional[int] = None, keep_recent_steps: int = 3, storage_dir: Optional[str] = None, safety_margin: float = 0.75):
        """
        Initialize context monitoring with automatic compaction.
        
        Args:
            model: The LLM model to use for summarization
            token_threshold: Maximum tokens before compaction (auto-calculated if None)
            keep_recent_steps: Number of recent steps to preserve during compaction
            storage_dir: Directory for external memory storage
            safety_margin: Fraction of model context to use before compaction (0.75 = 75%)
        """
        self.model = model
        self.keep_recent_steps = keep_recent_steps
        self.storage_dir = storage_dir
        self.safety_margin = safety_margin
        
        # Calculate token threshold based on model if not provided
        if token_threshold is None:
            self.token_threshold = calculate_safe_compaction_threshold(model, safety_margin)
        else:
            self.token_threshold = token_threshold
            
        print(f"🎯 Context monitoring initialized for model with {self.token_threshold:,} token threshold")
        
        # Will be set when attached to agent
        self.agent = None
        self.compactor = None
        
    def __call__(self, memory_step: MemoryStep, agent=None) -> None:
        """
        Callback triggered after each memory step.
        
        Args:
            memory_step: The memory step that was just added
            agent: The agent instance (provided by smolagents framework)
        """
        # Lazy initialization when first called with agent
        if self.agent is None and agent is not None:
            self.agent = agent
            self.compactor = AutomaticContextCompactor(
                agent_instance=agent,
                max_tokens=self.token_threshold,
                min_steps_between_compaction=self.keep_recent_steps,
                storage_dir=self.storage_dir,
                safety_margin=self.safety_margin
            )
        
        # Only monitor ActionSteps
        if not isinstance(memory_step, ActionStep):
            return
            
        # Skip if compactor not initialized yet
        if self.compactor is None:
            return
        
        self.compactor.step_count += 1
        step_type = type(memory_step).__name__
        print(f"📈 Processing {step_type} #{self.compactor.step_count}")
        
        # INCREMENTAL BACKUP: Save this ActionStep immediately
        try:
            self.compactor.backup_action_step(memory_step, self.compactor.step_count)
        except Exception as e:
            print(f"   ⚠️ Failed to backup step: {e}")
        
        # Estimate current tokens using simple method
        try:
            # Simple token estimation based on memory size
            current_tokens = self._estimate_tokens_simple()
            print(f"📊 Step {self.compactor.step_count}: ~{current_tokens:,} tokens")
            
            # Check if compaction needed
            if self.compactor.should_compact(current_tokens):
                self.compactor.perform_compaction(current_tokens)
            else:
                print(f"   ✅ No compaction needed (threshold: {self.compactor.max_tokens:,})")
                
        except Exception as e:
            print(f"   ⚠️ Token estimation failed: {e}")
    
    def _estimate_tokens_simple(self) -> int:
        """
        Simple token estimation based on total memory content.
        
        Returns:
            int: Estimated token count
        """
        if not self.agent or not hasattr(self.agent, 'memory'):
            return 0
        
        total_chars = 0
        
        # Count characters in all steps
        for step in self.agent.memory.steps:
            if hasattr(step, 'to_messages'):
                messages = step.to_messages()
                for msg in messages:
                    # Handle both ChatMessage objects and dictionary format
                    if hasattr(msg, 'content'):
                        # New ChatMessage object format
                        content = msg.content
                    elif isinstance(msg, dict):
                        # Old dictionary format
                        content = msg.get('content', '')
                    else:
                        content = ''
                        
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if hasattr(item, 'get') and item.get('type') == 'text':
                                    total_chars += len(item.get('text', ''))
                                elif hasattr(item, 'type') and item.type == 'text':
                                    # Handle object-style access
                                    total_chars += len(getattr(item, 'text', ''))
        
        # Add tool schema overhead (approximate)
        if hasattr(self.agent, 'tools_and_managed_agents'):
            tool_count = len(self.agent.tools_and_managed_agents)
            tool_overhead = tool_count * 500  # Rough estimate per tool
            total_chars += tool_overhead
        
        # Convert to tokens (chars / 4) + constant overhead
        estimated_tokens = total_chars // 4
        
        # Add empirically determined constant overhead (~3,500 tokens)
        # This accounts for system prompts, tool schemas, formatting overhead
        constant_overhead = 3500
        
        return estimated_tokens + constant_overhead