"""
YouTube Innovation Video Scraper.

Uses the YouTube Data API v3 (via httpx) and youtube-transcript-api to extract
innovation-related data from specified YouTube channels.

Requires:
- YOUTUBE_API_KEY environment variable
- youtube-transcript-api package

Workflow:
1. Search specified channels for innovation-related videos
2. Fetch video metadata (title, description, tags)
3. Pull transcripts via youtube-transcript-api
4. Combine metadata + transcript into RawCase objects
"""

from __future__ import annotations

import logging
import os
from typing import Any

from youtube_transcript_api import (
    YouTubeTranscriptApi,
)
from youtube_transcript_api.formatters import TextFormatter

from jugaad_bench.models import RawCase
from jugaad_bench.data.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# YouTube Data API v3 base URL
_YT_API_BASE = "https://www.googleapis.com/youtube/v3"

# Innovation-related search queries
_SEARCH_QUERIES: list[str] = [
    "jugaad innovation India",
    "grassroots innovation India",
    "frugal innovation rural India",
    "Indian farmer innovation",
    "village inventor India",
    "NIF innovation award",
    "Honey Bee Network innovation",
    "low cost innovation India",
    "desi jugaad technology",
    "rural innovation solution",
]


class YouTubeScraper(BaseScraper):
    """Scraper that extracts innovation data from YouTube videos.

    Args:
        channels: List of YouTube channel IDs/usernames to search.
        api_key: YouTube Data API v3 key. If None, reads YOUTUBE_API_KEY env var.
        max_videos_per_channel: Maximum videos to process per channel.
        max_search_results: Maximum total search results across queries.
        rate_limit_seconds: Seconds between YouTube API requests.
        transcript_languages: Preferred transcript languages (ordered).
    """

    def __init__(
        self,
        channels: list[str] | None = None,
        api_key: str | None = None,
        max_videos_per_channel: int = 100,
        max_search_results: int = 500,
        rate_limit_seconds: float = 0.5,
        transcript_languages: list[str] | None = None,
    ) -> None:
        super().__init__(
            source_name="youtube",
            rate_limit_seconds=rate_limit_seconds,
        )
        self.channels = channels or []
        self.max_videos_per_channel = max_videos_per_channel
        self.max_search_results = max_search_results
        self.transcript_languages = transcript_languages or ["en", "hi", "ta", "te", "kn"]

        # Resolve API key
        self._api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "YouTube Data API key required. Set the YOUTUBE_API_KEY "
                "environment variable or pass api_key to the constructor."
            )

        self._transcript_formatter = TextFormatter()

    async def scrape(self) -> list[RawCase]:
        """Execute the full YouTube scraping workflow.

        Returns:
            List of RawCase objects with video metadata + transcripts.
        """
        all_video_ids: set[str] = set()
        all_cases: list[RawCase] = []

        # Step 1: Collect video IDs from channel searches
        for channel_id in self.channels:
            logger.info("Searching channel: %s", channel_id)
            try:
                video_ids = await self._search_channel(channel_id)
                all_video_ids.update(video_ids)
                logger.info(
                    "Found %d videos in channel %s", len(video_ids), channel_id
                )
            except Exception:
                logger.exception("Error searching channel: %s", channel_id)

        # Step 2: Also run broad innovation-related searches
        for query in _SEARCH_QUERIES:
            if len(all_video_ids) >= self.max_search_results:
                break
            try:
                video_ids = await self._search_videos(query)
                new_ids = video_ids - all_video_ids
                all_video_ids.update(new_ids)
                logger.info(
                    "Search '%s': found %d new videos (total: %d)",
                    query,
                    len(new_ids),
                    len(all_video_ids),
                )
            except Exception:
                logger.exception("Error searching for: %s", query)

        logger.info("Total unique videos to process: %d", len(all_video_ids))

        # Step 3: Fetch metadata and transcripts for each video
        video_ids_list = sorted(all_video_ids)
        # Process in batches of 50 (YouTube API limit for videos.list)
        for batch_start in range(0, len(video_ids_list), 50):
            batch = video_ids_list[batch_start : batch_start + 50]
            try:
                metadata_map = await self._fetch_video_metadata(batch)
            except Exception:
                logger.exception("Error fetching metadata for batch")
                continue

            for video_id in batch:
                metadata = metadata_map.get(video_id)
                if metadata is None:
                    continue

                # Fetch transcript
                transcript_text = self._fetch_transcript(video_id)

                # Build RawCase
                raw_case = self._build_raw_case(video_id, metadata, transcript_text)
                if raw_case is not None:
                    all_cases.append(raw_case)

        logger.info("Total YouTube entries extracted: %d", len(all_cases))
        return all_cases

    # ------------------------------------------------------------------
    # YouTube API calls
    # ------------------------------------------------------------------

    async def _search_channel(self, channel_id: str) -> set[str]:
        """Search for innovation videos within a specific channel.

        Args:
            channel_id: YouTube channel ID or username.

        Returns:
            Set of video IDs found.
        """
        video_ids: set[str] = set()

        # First, resolve channel ID if it's a username
        resolved_channel_id = await self._resolve_channel_id(channel_id)
        if not resolved_channel_id:
            logger.warning("Could not resolve channel: %s", channel_id)
            return video_ids

        # Search the channel for innovation-related content
        page_token: str | None = None
        fetched = 0

        while fetched < self.max_videos_per_channel:
            params: dict[str, Any] = {
                "part": "snippet",
                "channelId": resolved_channel_id,
                "type": "video",
                "maxResults": 50,
                "order": "relevance",
                "q": "innovation OR jugaad OR invention",
                "key": self._api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            data = await self.fetch_json(
                f"{_YT_API_BASE}/search", params=params, use_cache=True
            )

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                vid_id = item.get("id", {}).get("videoId")
                if vid_id:
                    video_ids.add(vid_id)

            fetched += len(items)
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return video_ids

    async def _resolve_channel_id(self, channel_identifier: str) -> str | None:
        """Resolve a channel username or handle to a channel ID.

        If the identifier already looks like a channel ID (starts with UC),
        return it as-is. Otherwise, look it up via the channels API.
        """
        if channel_identifier.startswith("UC") and len(channel_identifier) == 24:
            return channel_identifier

        # Try forHandle lookup
        try:
            params: dict[str, Any] = {
                "part": "id",
                "forHandle": channel_identifier.lstrip("@"),
                "key": self._api_key,
            }
            data = await self.fetch_json(
                f"{_YT_API_BASE}/channels", params=params, use_cache=True
            )
            items = data.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

        # Try forUsername lookup
        try:
            params = {
                "part": "id",
                "forUsername": channel_identifier,
                "key": self._api_key,
            }
            data = await self.fetch_json(
                f"{_YT_API_BASE}/channels", params=params, use_cache=True
            )
            items = data.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

        # Try as a direct channel ID
        try:
            params = {
                "part": "id",
                "id": channel_identifier,
                "key": self._api_key,
            }
            data = await self.fetch_json(
                f"{_YT_API_BASE}/channels", params=params, use_cache=True
            )
            items = data.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

        return None

    async def _search_videos(self, query: str) -> set[str]:
        """Search YouTube for videos matching a query.

        Returns:
            Set of video IDs matching the query.
        """
        video_ids: set[str] = set()

        params: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "maxResults": 50,
            "order": "relevance",
            "q": query,
            "relevanceLanguage": "en",
            "key": self._api_key,
        }

        data = await self.fetch_json(
            f"{_YT_API_BASE}/search", params=params, use_cache=True
        )

        for item in data.get("items", []):
            vid_id = item.get("id", {}).get("videoId")
            if vid_id:
                video_ids.add(vid_id)

        return video_ids

    async def _fetch_video_metadata(
        self, video_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch detailed metadata for a batch of video IDs.

        Args:
            video_ids: List of video IDs (max 50).

        Returns:
            Dict mapping video_id -> metadata dict with keys:
            title, description, tags, channel_title, published_at.
        """
        params: dict[str, Any] = {
            "part": "snippet,contentDetails",
            "id": ",".join(video_ids),
            "key": self._api_key,
        }

        data = await self.fetch_json(
            f"{_YT_API_BASE}/videos", params=params, use_cache=True
        )

        result: dict[str, dict[str, Any]] = {}
        for item in data.get("items", []):
            vid_id = item["id"]
            snippet = item.get("snippet", {})
            result[vid_id] = {
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "tags": snippet.get("tags", []),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "category_id": snippet.get("categoryId", ""),
            }

        return result

    # ------------------------------------------------------------------
    # Transcript extraction
    # ------------------------------------------------------------------

    def _fetch_transcript(self, video_id: str) -> str | None:
        """Fetch the transcript for a video using youtube-transcript-api.

        Tries preferred languages in order, falling back to any available
        transcript.

        Args:
            video_id: YouTube video ID.

        Returns:
            Transcript text, or None if no transcript is available.
        """
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(
                video_id,
                languages=self.transcript_languages,
            )
            formatted = self._transcript_formatter.format_transcript(transcript)
            return formatted
        except Exception:
            # Try to find any available transcript
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.list(video_id)
                # Get the first available transcript
                for t in transcript_list:
                    try:
                        transcript = t.fetch()
                        formatted = self._transcript_formatter.format_transcript(transcript)
                        return formatted
                    except Exception:
                        continue
            except Exception:
                logger.debug("No transcript available for video: %s", video_id)
                return None
        return None

    # ------------------------------------------------------------------
    # RawCase construction
    # ------------------------------------------------------------------

    def _build_raw_case(
        self,
        video_id: str,
        metadata: dict[str, Any],
        transcript_text: str | None,
    ) -> RawCase | None:
        """Build a RawCase from video metadata and transcript.

        Args:
            video_id: YouTube video ID.
            metadata: Video metadata dict from the API.
            transcript_text: Full transcript text, or None.

        Returns:
            RawCase or None if there's insufficient content.
        """
        title = metadata.get("title", "").strip()
        description = metadata.get("description", "").strip()
        tags = metadata.get("tags", [])
        channel = metadata.get("channel_title", "")

        # Assemble the raw text
        text_parts: list[str] = []
        if title:
            text_parts.append(f"Title: {title}")
        if description:
            text_parts.append(f"Description: {description}")
        if tags:
            text_parts.append(f"Tags: {', '.join(tags[:20])}")
        if channel:
            text_parts.append(f"Channel: {channel}")
        if transcript_text:
            # Limit transcript to ~5000 chars to keep entries manageable
            truncated = transcript_text[:5000]
            if len(transcript_text) > 5000:
                truncated += "\n[... transcript truncated ...]"
            text_parts.append(f"Transcript:\n{truncated}")

        raw_text = "\n\n".join(text_parts)

        # Require minimum content length
        if len(raw_text) < 20:
            return None

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            return RawCase(
                source="youtube",
                url_or_path=video_url,
                raw_text=raw_text,
                title=title or None,
                innovator_name=None,
                location=None,
                category=None,
            )
        except Exception:
            logger.warning("Failed to create RawCase for video: %s", video_id)
            return None
