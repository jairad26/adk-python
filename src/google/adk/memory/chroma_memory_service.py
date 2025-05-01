# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import chromadb
from typing_extensions import override

from ..events.event import Event
from ..sessions.session import Session
from .base_memory_service import BaseMemoryService
from .base_memory_service import MemoryResult
from .base_memory_service import SearchMemoryResponse
from google.genai import types
from ..events.event_actions import EventActions

class ChromaMemoryService(BaseMemoryService):
    """A memory service that uses Chroma for storage and retrieval."""
    def __init__(self, top_k: int = 10):
        """Initializes a ChromaMemoryService.
        Args:
            collection_name: The name of the Chroma collection to use.
        """
        self.client = chromadb.EphemeralClient()
        self.top_k = top_k
    @override
    def add_session_to_memory(self, session: Session):
        """Adds a session to the memory service.
        
        Args:
            session: The session to add.
        """
        collection = self.client.get_or_create_collection(f"{session.app_name}_{session.user_id}")
        
        for event in session.events:
            if not event.content or not event.content.parts:
                continue
                
            text_parts = []
            for part in event.content.parts:
                if part.text:
                    text_parts.append(part.text)
            
            if not text_parts:
                continue
                
            text = " ".join(text_parts)
            
            collection.add(
                ids=[f"{session.id}_{event.id}"],
                documents=[text],
                metadatas=[{
                    "session_id": session.id,
                    "event_id": event.id,
                    "invocation_id": event.invocation_id,
                    "author": event.author,
                    "timestamp": event.timestamp,
                    "branch": event.branch or "",
                    "actions": event.actions.model_dump_json() if event.actions else None,
                    "long_running_tool_ids": str(list(event.long_running_tool_ids)) if event.long_running_tool_ids else None,
                    "grounding_metadata": event.grounding_metadata.model_dump_json() if event.grounding_metadata else None,
                    "partial": event.partial,
                    "turn_complete": event.turn_complete,
                    "error_code": event.error_code,
                    "error_message": event.error_message,
                    "interrupted": event.interrupted,
                    "custom_metadata": event.custom_metadata,
                }],
            )

    @override
    def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Searches for sessions that match the query using both semantic and keyword search.

        Args:
            app_name: The name of the application.
            user_id: The id of the user.
            query: The query to search for.

        Returns:
            A SearchMemoryResponse containing the matching memories.
        """
        try:
            collection = self.client.get_collection(f"{app_name}_{user_id}")
        except Exception as e:
            return SearchMemoryResponse(memories=[])
        
        # Perform semantic search
        semantic_results = collection.query(
            query_texts=[query],
            n_results=self.top_k,
        )
        
        # Perform keyword search
        keywords = set(query.lower().split())
        keyword_results = collection.query(
            query_texts=[query],
            n_results=self.top_k,
            where_document={"$or": [
                {"$contains": keyword}
                for keyword in keywords
            ]}
        )
        
        session_events = {}
        for results in [semantic_results, keyword_results]:
            for i, doc_id in enumerate(results["ids"][0]):
                session_id = doc_id.split("_")[0]
                event_id = doc_id.split("_")[1]
                
                    
                metadata = results["metadatas"][0][i]
                event = Event(
                    id=event_id,
                    invocation_id=metadata["invocation_id"],
                    author=metadata["author"],
                    timestamp=metadata["timestamp"],
                    content=types.Content(parts=[types.Part(text=results["documents"][0][i])]),
                    branch=metadata["branch"] if metadata["branch"] else None,
                    actions=EventActions.model_validate_json(metadata["actions"]) if metadata["actions"] else None,
                    long_running_tool_ids=set(eval(metadata["long_running_tool_ids"])) if metadata["long_running_tool_ids"] else None,
                    grounding_metadata=types.GroundingMetadata.model_validate_json(metadata["grounding_metadata"]) if metadata["grounding_metadata"] else None,
                    partial=metadata["partial"],
                    turn_complete=metadata["turn_complete"],
                    error_code=metadata["error_code"],
                    error_message=metadata["error_message"],
                    interrupted=metadata["interrupted"],
                    custom_metadata=metadata["custom_metadata"]
                )
                
                if session_id not in session_events:
                    session_events[session_id] = []
                session_events[session_id].append(event)
        
        # Create memory results
        memory_results = []
        for session_id, events in session_events.items():
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            memory_results.append(
                MemoryResult(session_id=session_id, events=sorted_events)
            )
        
        return SearchMemoryResponse(memories=memory_results)
        
      
