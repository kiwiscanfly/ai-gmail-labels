"""Orchestration agent using LangGraph for workflow management."""

import asyncio
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import structlog

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.core.state_manager import get_state_manager
from src.core.event_bus import get_event_bus
from src.core.error_recovery import get_error_recovery_manager
from src.models.agent import AgentMessage, WorkflowCheckpoint
from src.models.email import EmailMessage, EmailCategory
from src.models.common import Status, Priority, MessageType
from src.core.exceptions import AgentError, WorkflowError

logger = structlog.get_logger(__name__)


@dataclass
class EmailWorkflowState:
    """State for email processing workflow."""
    # Core data
    emails: List[EmailMessage] = field(default_factory=list)
    processed_emails: List[str] = field(default_factory=list)
    failed_emails: List[str] = field(default_factory=list)
    categorizations: Dict[str, EmailCategory] = field(default_factory=dict)
    
    # Workflow control
    current_step: str = "start"
    workflow_id: str = ""
    started_at: float = 0.0
    
    # Agent states
    retrieval_complete: bool = False
    categorization_complete: bool = False
    user_interactions_pending: List[str] = field(default_factory=list)
    
    # Results
    total_processed: int = 0
    total_failed: int = 0
    labels_applied: int = 0
    
    # Messages for LangGraph
    messages: List[BaseMessage] = field(default_factory=list)
    
    def add_message(self, content: str, message_type: str = "human") -> None:
        """Add a message to the workflow state."""
        if message_type == "ai":
            self.messages.append(AIMessage(content=content))
        else:
            self.messages.append(HumanMessage(content=content))


class OrchestrationAgent:
    """Main orchestration agent that coordinates email processing workflow."""
    
    def __init__(self):
        self._state_manager = None
        self._event_bus = None
        self._error_manager = None
        self._workflow_graph = None
        self._checkpointer = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the orchestration agent."""
        try:
            self._state_manager = await get_state_manager()
            self._event_bus = await get_event_bus()
            self._error_manager = await get_error_recovery_manager()
            
            # Initialize LangGraph workflow
            await self._setup_workflow_graph()
            
            # Subscribe to relevant events
            await self._event_bus.subscribe("orchestration_agent", self._handle_agent_message)
            
            self._initialized = True
            logger.info("Orchestration agent initialized")
            
        except Exception as e:
            logger.error("Failed to initialize orchestration agent", error=str(e))
            raise AgentError(f"Failed to initialize orchestration agent: {e}")
    
    async def _setup_workflow_graph(self) -> None:
        """Set up the LangGraph workflow for email processing."""
        try:
            # Create SQLite checkpointer for workflow persistence
            from pathlib import Path
            checkpoint_db = Path("data/workflow_checkpoints.db")
            checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
            
            self._checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_db}")
            
            # Define the workflow graph
            workflow = StateGraph(EmailWorkflowState)
            
            # Add nodes for each step
            workflow.add_node("retrieve_emails", self._retrieve_emails_node)
            workflow.add_node("categorize_emails", self._categorize_emails_node)
            workflow.add_node("handle_user_interactions", self._handle_user_interactions_node)
            workflow.add_node("apply_labels", self._apply_labels_node)
            workflow.add_node("finalize_workflow", self._finalize_workflow_node)
            
            # Define the workflow edges
            workflow.set_entry_point("retrieve_emails")
            
            # Linear flow with conditional branching
            workflow.add_edge("retrieve_emails", "categorize_emails")
            workflow.add_conditional_edges(
                "categorize_emails",
                self._should_handle_user_interactions,
                {
                    "user_interactions": "handle_user_interactions",
                    "apply_labels": "apply_labels"
                }
            )
            workflow.add_edge("handle_user_interactions", "apply_labels")
            workflow.add_edge("apply_labels", "finalize_workflow")
            workflow.add_edge("finalize_workflow", END)
            
            # Compile the graph with checkpointing
            self._workflow_graph = workflow.compile(checkpointer=self._checkpointer)
            
            logger.info("LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error("Failed to setup workflow graph", error=str(e))
            raise WorkflowError(f"Failed to setup workflow graph: {e}")
    
    async def start_email_processing_workflow(
        self, 
        emails: List[EmailMessage],
        workflow_id: Optional[str] = None
    ) -> str:
        """Start a new email processing workflow."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate workflow ID if not provided
            if not workflow_id:
                workflow_id = f"email_workflow_{datetime.now().timestamp()}"
            
            # Create initial state
            initial_state = EmailWorkflowState(
                emails=emails,
                workflow_id=workflow_id,
                started_at=datetime.now().timestamp(),
                current_step="start"
            )
            
            initial_state.add_message(
                f"Starting email processing workflow for {len(emails)} emails",
                "human"
            )
            
            # Create workflow configuration
            config = {
                "configurable": {"thread_id": workflow_id},
                "recursion_limit": 50
            }
            
            # Start the workflow
            logger.info(
                "Starting email processing workflow",
                workflow_id=workflow_id,
                email_count=len(emails)
            )
            
            # Run the workflow asynchronously
            asyncio.create_task(self._execute_workflow(initial_state, config))
            
            return workflow_id
            
        except Exception as e:
            logger.error(
                "Failed to start email processing workflow",
                workflow_id=workflow_id,
                error=str(e)
            )
            raise WorkflowError(f"Failed to start workflow: {e}")
    
    async def _execute_workflow(self, initial_state: EmailWorkflowState, config: Dict) -> None:
        """Execute the workflow with error handling."""
        try:
            async with self._error_manager.protected_operation(
                f"workflow_{initial_state.workflow_id}",
                "orchestration_agent"
            ):
                # Execute the workflow
                final_state = None
                async for state in self._workflow_graph.astream(initial_state, config):
                    final_state = state
                    
                    # Log progress
                    current_step = state.get("current_step", "unknown")
                    logger.debug(
                        "Workflow step completed",
                        workflow_id=initial_state.workflow_id,
                        step=current_step
                    )
                
                # Save final checkpoint
                if final_state:
                    await self._save_workflow_checkpoint(final_state)
                    
                logger.info(
                    "Workflow completed successfully",
                    workflow_id=initial_state.workflow_id,
                    processed=final_state.get("total_processed", 0),
                    failed=final_state.get("total_failed", 0)
                )
                
        except Exception as e:
            logger.error(
                "Workflow execution failed",
                workflow_id=initial_state.workflow_id,
                error=str(e)
            )
            # Save error state
            await self._save_workflow_error(initial_state.workflow_id, str(e))
    
    # Workflow node implementations
    
    async def _retrieve_emails_node(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Node for retrieving emails."""
        state.current_step = "retrieve_emails"
        state.add_message("Retrieving emails from Gmail", "ai")
        
        try:
            # Send message to email retrieval agent
            message = AgentMessage(
                sender_agent="orchestration_agent",
                recipient_agent="email_retrieval_agent",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "retrieve_emails",
                    "workflow_id": state.workflow_id,
                    "email_ids": [email.id for email in state.emails]
                },
                priority=Priority.HIGH
            )
            
            await self._event_bus.publish(message)
            
            # Mark retrieval as initiated
            state.retrieval_complete = True
            state.add_message(f"Email retrieval initiated for {len(state.emails)} emails", "ai")
            
            logger.info(
                "Email retrieval node completed",
                workflow_id=state.workflow_id,
                email_count=len(state.emails)
            )
            
        except Exception as e:
            logger.error("Email retrieval node failed", workflow_id=state.workflow_id, error=str(e))
            state.add_message(f"Email retrieval failed: {e}", "ai")
            raise
        
        return state
    
    async def _categorize_emails_node(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Node for categorizing emails."""
        state.current_step = "categorize_emails"
        state.add_message("Categorizing emails using AI", "ai")
        
        try:
            # Send message to categorization agent
            message = AgentMessage(
                sender_agent="orchestration_agent",
                recipient_agent="categorization_agent",
                message_type=MessageType.COMMAND,
                payload={
                    "command": "categorize_emails",
                    "workflow_id": state.workflow_id,
                    "emails": [email.to_dict() for email in state.emails]
                },
                priority=Priority.HIGH
            )
            
            await self._event_bus.publish(message)
            
            # For now, simulate categorization results
            # In real implementation, this would wait for agent response
            for email in state.emails:
                state.categorizations[email.id] = EmailCategory(
                    email_id=email.id,
                    suggested_labels=["Important", "Work"],
                    confidence_scores={"Important": 0.85, "Work": 0.75},
                    reasoning="AI analysis of email content",
                    requires_user_input=False
                )
            
            state.categorization_complete = True
            state.add_message(f"Categorized {len(state.emails)} emails", "ai")
            
            logger.info(
                "Email categorization node completed",
                workflow_id=state.workflow_id,
                categorized_count=len(state.categorizations)
            )
            
        except Exception as e:
            logger.error("Email categorization node failed", workflow_id=state.workflow_id, error=str(e))
            state.add_message(f"Email categorization failed: {e}", "ai")
            raise
        
        return state
    
    async def _handle_user_interactions_node(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Node for handling user interactions."""
        state.current_step = "handle_user_interactions"
        state.add_message("Handling user interactions for ambiguous emails", "ai")
        
        try:
            # Identify emails requiring user input
            needs_input = []
            for email_id, category in state.categorizations.items():
                if category.requires_user_input:
                    needs_input.append(email_id)
            
            if needs_input:
                # Send message to user interaction agent
                message = AgentMessage(
                    sender_agent="orchestration_agent",
                    recipient_agent="user_interaction_agent",
                    message_type=MessageType.COMMAND,
                    payload={
                        "command": "handle_interactions",
                        "workflow_id": state.workflow_id,
                        "email_ids": needs_input,
                        "categorizations": {eid: cat.to_dict() for eid, cat in state.categorizations.items() if eid in needs_input}
                    },
                    priority=Priority.MEDIUM
                )
                
                await self._event_bus.publish(message)
                state.user_interactions_pending = needs_input
                state.add_message(f"User input requested for {len(needs_input)} emails", "ai")
            else:
                state.add_message("No user interactions required", "ai")
            
            logger.info(
                "User interactions node completed",
                workflow_id=state.workflow_id,
                interactions_needed=len(needs_input)
            )
            
        except Exception as e:
            logger.error("User interactions node failed", workflow_id=state.workflow_id, error=str(e))
            state.add_message(f"User interactions handling failed: {e}", "ai")
            raise
        
        return state
    
    async def _apply_labels_node(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Node for applying labels to emails."""
        state.current_step = "apply_labels"
        state.add_message("Applying labels to categorized emails", "ai")
        
        try:
            applied_count = 0
            
            for email in state.emails:
                category = state.categorizations.get(email.id)
                if category and category.suggested_labels:
                    # Apply labels via Gmail API
                    # This would be handled by the Gmail integration
                    state.processed_emails.append(email.id)
                    applied_count += 1
                else:
                    state.failed_emails.append(email.id)
            
            state.labels_applied = applied_count
            state.total_processed = len(state.processed_emails)
            state.total_failed = len(state.failed_emails)
            
            state.add_message(f"Applied labels to {applied_count} emails", "ai")
            
            logger.info(
                "Label application node completed",
                workflow_id=state.workflow_id,
                applied=applied_count,
                failed=len(state.failed_emails)
            )
            
        except Exception as e:
            logger.error("Label application node failed", workflow_id=state.workflow_id, error=str(e))
            state.add_message(f"Label application failed: {e}", "ai")
            raise
        
        return state
    
    async def _finalize_workflow_node(self, state: EmailWorkflowState) -> EmailWorkflowState:
        """Node for finalizing the workflow."""
        state.current_step = "finalize_workflow"
        
        try:
            # Generate final summary
            summary = {
                "workflow_id": state.workflow_id,
                "total_emails": len(state.emails),
                "processed": state.total_processed,
                "failed": state.total_failed,
                "labels_applied": state.labels_applied,
                "duration": datetime.now().timestamp() - state.started_at
            }
            
            state.add_message(
                f"Workflow completed: {state.total_processed} processed, {state.total_failed} failed",
                "ai"
            )
            
            # Save workflow results
            await self._save_workflow_results(state, summary)
            
            logger.info(
                "Workflow finalized",
                workflow_id=state.workflow_id,
                summary=summary
            )
            
        except Exception as e:
            logger.error("Workflow finalization failed", workflow_id=state.workflow_id, error=str(e))
            state.add_message(f"Workflow finalization failed: {e}", "ai")
            raise
        
        return state
    
    def _should_handle_user_interactions(self, state: EmailWorkflowState) -> Literal["user_interactions", "apply_labels"]:
        """Conditional edge function to determine if user interactions are needed."""
        needs_input = any(
            cat.requires_user_input 
            for cat in state.categorizations.values()
        )
        
        if needs_input:
            return "user_interactions"
        else:
            return "apply_labels"
    
    # Helper methods
    
    async def _save_workflow_checkpoint(self, state: EmailWorkflowState) -> None:
        """Save workflow checkpoint to state manager."""
        try:
            checkpoint = WorkflowCheckpoint(
                workflow_id=state.workflow_id,
                workflow_type="email_processing",
                state_data={
                    "current_step": state.current_step,
                    "processed_emails": state.processed_emails,
                    "failed_emails": state.failed_emails,
                    "total_processed": state.total_processed,
                    "total_failed": state.total_failed,
                    "labels_applied": state.labels_applied
                },
                checkpoint_time=datetime.now().timestamp(),
                status=Status.IN_PROGRESS.value
            )
            
            await self._state_manager.save_checkpoint(checkpoint)
            
        except Exception as e:
            logger.error("Failed to save workflow checkpoint", workflow_id=state.workflow_id, error=str(e))
    
    async def _save_workflow_error(self, workflow_id: str, error_message: str) -> None:
        """Save workflow error state."""
        try:
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_id,
                workflow_type="email_processing",
                state_data={"error": error_message},
                checkpoint_time=datetime.now().timestamp(),
                status=Status.FAILED.value,
                error_message=error_message
            )
            
            await self._state_manager.save_checkpoint(checkpoint)
            
        except Exception as e:
            logger.error("Failed to save workflow error", workflow_id=workflow_id, error=str(e))
    
    async def _save_workflow_results(self, state: EmailWorkflowState, summary: Dict[str, Any]) -> None:
        """Save final workflow results."""
        try:
            checkpoint = WorkflowCheckpoint(
                workflow_id=state.workflow_id,
                workflow_type="email_processing",
                state_data=summary,
                checkpoint_time=datetime.now().timestamp(),
                status=Status.COMPLETED.value
            )
            
            await self._state_manager.save_checkpoint(checkpoint)
            
            # Publish completion event
            completion_message = AgentMessage(
                sender_agent="orchestration_agent",
                recipient_agent="all",
                message_type=MessageType.EVENT,
                payload={
                    "event": "workflow_completed",
                    "workflow_id": state.workflow_id,
                    "summary": summary
                },
                priority=Priority.LOW
            )
            
            await self._event_bus.publish(completion_message)
            
        except Exception as e:
            logger.error("Failed to save workflow results", workflow_id=state.workflow_id, error=str(e))
    
    async def _handle_agent_message(self, message: AgentMessage) -> None:
        """Handle incoming messages from other agents."""
        try:
            if message.message_type == MessageType.RESPONSE:
                # Handle agent responses
                payload = message.payload
                
                if payload.get("command") == "retrieve_emails":
                    logger.info("Email retrieval completed", workflow_id=payload.get("workflow_id"))
                elif payload.get("command") == "categorize_emails":
                    logger.info("Email categorization completed", workflow_id=payload.get("workflow_id"))
                elif payload.get("command") == "handle_interactions":
                    logger.info("User interactions completed", workflow_id=payload.get("workflow_id"))
            
        except Exception as e:
            logger.error("Failed to handle agent message", message_id=message.id, error=str(e))
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        try:
            checkpoint = await self._state_manager.load_checkpoint(workflow_id)
            if checkpoint:
                return {
                    "workflow_id": checkpoint.workflow_id,
                    "status": checkpoint.status,
                    "current_step": checkpoint.state_data.get("current_step"),
                    "processed": checkpoint.state_data.get("total_processed", 0),
                    "failed": checkpoint.state_data.get("total_failed", 0),
                    "last_updated": checkpoint.checkpoint_time
                }
            return None
            
        except Exception as e:
            logger.error("Failed to get workflow status", workflow_id=workflow_id, error=str(e))
            return None


# Global orchestration agent instance
_orchestration_agent: Optional[OrchestrationAgent] = None


async def get_orchestration_agent() -> OrchestrationAgent:
    """Get the global orchestration agent instance."""
    global _orchestration_agent
    if _orchestration_agent is None:
        _orchestration_agent = OrchestrationAgent()
        await _orchestration_agent.initialize()
    return _orchestration_agent