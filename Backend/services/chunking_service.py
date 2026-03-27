import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple


from Backend.database.models import ConversationThread, ThreadChunk
from Backend.llm.llm_provider import get_node_llm_provider

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(self):
        try:
            llm = get_node_llm_provider("chunking")
        except Exception as exc:
            logger.error(
                "Failed to initialize LLM provider for chunking: %s", exc)
            raise RuntimeError(
                "ChunkingService initialization failed: unable to obtain LLM provider for 'chunking'"
            ) from exc

        if llm is None:
            error_message = (
                "ChunkingService initialization failed: get_node_llm_provider('chunking') returned None"
            )
            logger.error(error_message)
            raise RuntimeError(error_message)

        self.llm = llm
        self._max_chunk_size = 500  # Words or basic text length approximations if needed

    async def summarize_thread(self, text: str) -> str:
        prompt = f"Summarize the following social media conversation thread. Capture the main topic, participant viewpoints, and any significant conclusions or facts mentioned:\n\n{text}"
        try:
            summary = await self.llm.generate(prompt=prompt, system="You are an expert summarizer for social media threads.")
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to summarize thread: {e}")
            return "No summary available due to error."

    async def process_post_with_comments(self, post: Dict[str, Any], comments: List[Dict[str, Any]]) -> Tuple[ConversationThread, List[ThreadChunk]]:
        """
        Build a conversation thread and associated chunks from a single post and its comments.

        The method normalizes the post and comment data into a list of message dictionaries,
        infers the set of participants, creates a text representation of the full thread for
        summarization, then constructs a ConversationThread model and splits the thread into
        smaller ThreadChunk instances.

        Args:
            post: A dictionary representing the original social media post. Typically contains
                keys like "account_username", "content", and "created_at".
            comments: A list of dictionaries representing comments on the post. Each comment
                is expected to contain fields such as "commenter_name", "content", and "time".

        Returns:
            A tuple of:
                - ConversationThread: The constructed thread object with participants, messages,
                  and an LLM-generated summary.
                - List[ThreadChunk]: A list of chunks derived from the full thread text for
                  downstream processing or storage.
        """
        participants = set()
        if post.get("account_username"):
            participants.add(post["account_username"])

        messages = [
            {
                "role": "post",
                "author": post.get("account_username", ""),
                "content": post.get("content", ""),
                "time": post.get("created_at", "")
            }
        ]

        # Sort comments by time if possible
        comments_sorted = sorted(
            comments, key=lambda x: str(x.get("time", "")))

        for c in comments_sorted:
            author = c.get("commenter_name", "")
            if author:
                participants.add(author)
            messages.append({
                "role": "comment",
                "author": author,
                "content": c.get("content", ""),
                "time": c.get("time", "")
            })

        # Format thread for summarization
        thread_text_parts = []
        for msg in messages:
            thread_text_parts.append(
                f"{msg['author']} [{msg['time']}]: {msg['content']}")

        full_text = "\n".join(thread_text_parts)
        summary = await self.summarize_thread(full_text)

        thread_id = f"thread_post_{post.get('id', '')}"
        thread = ConversationThread(
            id=thread_id,
            source_type="post",
            participants=list(participants),
            messages=messages,
            summary=summary,
            created_at=post.get("created_at", ""),
            updated_at=messages[-1].get("time",
                                        "") if messages else post.get("created_at", ""),
        )

        chunks = self._chunk_thread(thread, full_text)
        return thread, chunks

    async def process_dm_thread(self, conversation_id: str, messages: List[Dict[str, Any]]) -> Tuple[ConversationThread, List[ThreadChunk]]:
        participants = set()
        dm_messages = []

        messages_sorted = sorted(messages, key=self._get_time_ms_for_sort)
        for msg in messages_sorted:
            sender = msg.get("sender_name", "")
            receiver = msg.get("receiver_name", "")
            if sender:
                participants.add(sender)
            if receiver:
                participants.add(receiver)

            dm_messages.append(
                {
                    "role": "message",
                    "author": sender,
                    "receiver": receiver,
                    "content": msg.get("text", ""),
                    "time": msg.get("date", "") or str(msg.get("time_ms", "")),
                }
            )

        thread_text_parts = []
        for msg in dm_messages:
            thread_text_parts.append(
                f"{msg['author']} -> {msg.get('receiver', '')} [{msg['time']}]: {msg['content']}"
            )

        full_text = "\n".join(thread_text_parts)
        summary = await self.summarize_thread(full_text)

        thread_id = f"thread_dm_{conversation_id}"
        thread = ConversationThread(
            id=thread_id,
            source_type="message",
            participants=list(participants),
            messages=dm_messages,
            summary=summary,
            created_at=dm_messages[0].get("time", "") if dm_messages else "",
            updated_at=dm_messages[-1].get("time", "") if dm_messages else "",
        )

        chunks = self._chunk_thread(thread, full_text)
        return thread, chunks

    @staticmethod
    def _get_time_ms_for_sort(message: Dict[str, Any]) -> int:
        raw = message.get("time_ms", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            date_str = str(message.get("date", "") or "")
            if not date_str:
                return 0
            try:
                return int(datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp() * 1000)
            except ValueError:
                return 0

    def _chunk_thread(self, thread: ConversationThread, full_text: str) -> List[ThreadChunk]:
        chunks = []

        # 1. Add summary as the parent chunk
        if thread.summary:
            chunks.append(ThreadChunk(
                id=f"{thread.id}_chunk_summary",
                thread_id=thread.id,
                content=f"Summary of conversation involving {', '.join(thread.participants)}: {thread.summary}",
                chunk_index=0,
                is_summary=True
            ))

        # 2. Add full text as a chunk if it's small enough, otherwise split it
        # Simple splitting by line for now
        lines = full_text.split("\n")
        current_chunk_lines = []
        current_length = 0
        chunk_index = 1

        for line in lines:
            line_len = len(line.split())
            if current_length + line_len > self._max_chunk_size and current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(ThreadChunk(
                    id=f"{thread.id}_chunk_{chunk_index}",
                    thread_id=thread.id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    is_summary=False
                ))
                chunk_index += 1
                current_chunk_lines = []
                current_length = 0

            current_chunk_lines.append(line)
            current_length += line_len

        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(ThreadChunk(
                id=f"{thread.id}_chunk_{chunk_index}",
                thread_id=thread.id,
                content=chunk_content,
                chunk_index=chunk_index,
                is_summary=False
            ))
        return chunks
