#!/usr/bin/env python3
"""
QNTI Social Trading System
Implements social trading feed with real-time updates, trader interactions, and community features
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import uuid4
import hashlib

logger = logging.getLogger('QNTI_SOCIAL_TRADING')

class PostType(Enum):
    """Social post types"""
    TRADE_SIGNAL = "trade_signal"
    MARKET_ANALYSIS = "market_analysis"
    TRADE_RESULT = "trade_result"
    STRATEGY_DISCUSSION = "strategy_discussion"
    NEWS_SHARE = "news_share"
    QUESTION = "question"
    EDUCATIONAL = "educational"
    GENERAL = "general"

class ReactionType(Enum):
    """Reaction types for posts"""
    LIKE = "like"
    LOVE = "love"
    AGREE = "agree"
    DISAGREE = "disagree"
    LAUGH = "laugh"
    BULLISH = "bullish"
    BEARISH = "bearish"

class NotificationType(Enum):
    """Notification types"""
    NEW_FOLLOWER = "new_follower"
    POST_LIKE = "post_like"
    POST_COMMENT = "post_comment"
    TRADE_SIGNAL = "trade_signal"
    MENTION = "mention"
    COPY_TRADE = "copy_trade"
    ACHIEVEMENT = "achievement"

class UserRole(Enum):
    """User roles in social trading"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    VERIFIED = "verified"
    MODERATOR = "moderator"

@dataclass
class SocialPost:
    """Social trading post"""
    id: str
    author_id: str
    author_name: str
    post_type: PostType
    content: str
    
    # Trade-specific fields
    symbol: Optional[str] = None
    signal_type: Optional[str] = None  # BUY/SELL
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: Optional[int] = None  # 1-10
    
    # Media attachments
    images: List[str] = None
    charts: List[str] = None
    
    # Engagement metrics
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0
    
    # Reactions
    reactions: Dict[str, int] = None  # reaction_type -> count
    
    # Metadata
    tags: List[str] = None
    mentioned_users: List[str] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    
    # Moderation
    is_pinned: bool = False
    is_featured: bool = False
    is_deleted: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.images is None:
            self.images = []
        if self.charts is None:
            self.charts = []
        if self.reactions is None:
            self.reactions = {}
        if self.tags is None:
            self.tags = []
        if self.mentioned_users is None:
            self.mentioned_users = []

@dataclass
class SocialComment:
    """Comment on a social post"""
    id: str
    post_id: str
    author_id: str
    author_name: str
    content: str
    
    # Nested comments
    parent_comment_id: Optional[str] = None
    replies: List[str] = None
    
    # Engagement
    likes: int = 0
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    # Moderation
    is_deleted: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.replies is None:
            self.replies = []

@dataclass
class SocialUser:
    """Social trading user profile"""
    id: str
    username: str
    display_name: str
    email: str
    role: UserRole
    
    # Profile information
    bio: str = ""
    location: str = ""
    avatar_url: str = ""
    cover_url: str = ""
    
    # Trading stats
    total_posts: int = 0
    total_signals: int = 0
    signal_accuracy: float = 0.0
    followers_count: int = 0
    following_count: int = 0
    
    # Engagement stats
    total_likes_received: int = 0
    total_comments_received: int = 0
    reputation_score: int = 0
    
    # Preferences
    notification_settings: Dict = None
    privacy_settings: Dict = None
    
    # Following/followers
    following: Set[str] = None
    followers: Set[str] = None
    blocked_users: Set[str] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_active: datetime = None
    
    # Verification
    is_verified: bool = False
    verification_level: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()
        if self.notification_settings is None:
            self.notification_settings = {
                'new_follower': True,
                'post_like': True,
                'post_comment': True,
                'trade_signal': True,
                'mention': True
            }
        if self.privacy_settings is None:
            self.privacy_settings = {
                'profile_visibility': 'public',
                'trade_history_visibility': 'followers',
                'allow_direct_messages': True
            }
        if self.following is None:
            self.following = set()
        if self.followers is None:
            self.followers = set()
        if self.blocked_users is None:
            self.blocked_users = set()

@dataclass
class SocialNotification:
    """Social trading notification"""
    id: str
    user_id: str
    notification_type: NotificationType
    title: str
    message: str
    
    # Related objects
    related_post_id: Optional[str] = None
    related_user_id: Optional[str] = None
    related_comment_id: Optional[str] = None
    
    # Metadata
    data: Dict = None
    
    # State
    is_read: bool = False
    is_deleted: bool = False
    
    # Timestamps
    created_at: datetime = None
    read_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.data is None:
            self.data = {}

@dataclass
class TradingRoom:
    """Trading room for group discussions"""
    id: str
    name: str
    description: str
    creator_id: str
    
    # Room settings
    is_public: bool = True
    requires_approval: bool = False
    max_members: int = 1000
    
    # Members
    members: Set[str] = None
    moderators: Set[str] = None
    banned_users: Set[str] = None
    
    # Stats
    member_count: int = 0
    post_count: int = 0
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.members is None:
            self.members = set()
        if self.moderators is None:
            self.moderators = set()
        if self.banned_users is None:
            self.banned_users = set()

class QNTISocialTrading:
    """QNTI Social Trading System"""
    
    def __init__(self, trade_manager, mt5_bridge=None, copy_trading=None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        self.copy_trading = copy_trading
        
        # Data storage
        self.users: Dict[str, SocialUser] = {}
        self.posts: Dict[str, SocialPost] = {}
        self.comments: Dict[str, SocialComment] = {}
        self.notifications: Dict[str, List[SocialNotification]] = {}
        self.trading_rooms: Dict[str, TradingRoom] = {}
        
        # Real-time connections
        self.active_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.websocket_callbacks: List[Callable] = []
        
        # Feed algorithms
        self.feed_algorithms = {
            'chronological': self._get_chronological_feed,
            'trending': self._get_trending_feed,
            'personalized': self._get_personalized_feed,
            'following': self._get_following_feed
        }
        
        # Content moderation
        self.banned_words = set()
        self.auto_moderation_enabled = True
        
        # Performance tracking
        self.performance_stats = {
            'total_posts': 0,
            'total_comments': 0,
            'total_likes': 0,
            'total_users': 0,
            'daily_active_users': 0,
            'signal_accuracy': 0.0
        }
        
        # Background tasks
        self.background_tasks_active = False
        self.background_thread = None
        
        # Load existing data
        self._load_data()
        
        # Start background tasks
        self.start_background_tasks()
        
        logger.info("Social Trading System initialized")
    
    def start_background_tasks(self):
        """Start background tasks"""
        if not self.background_tasks_active:
            self.background_tasks_active = True
            self.background_thread = threading.Thread(target=self._background_loop, daemon=True)
            self.background_thread.start()
            logger.info("Social trading background tasks started")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.background_tasks_active = False
        if self.background_thread:
            self.background_thread.join(timeout=1.0)
        logger.info("Social trading background tasks stopped")
    
    def _background_loop(self):
        """Background tasks loop"""
        while self.background_tasks_active:
            try:
                self._cleanup_expired_posts()
                self._update_user_activity()
                self._calculate_signal_accuracy()
                self._update_trending_posts()
                self._clean_old_notifications()
                time.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Error in social trading background loop: {e}")
                time.sleep(60)
    
    def _cleanup_expired_posts(self):
        """Clean up expired posts"""
        try:
            current_time = datetime.now()
            expired_posts = []
            
            for post_id, post in self.posts.items():
                if post.expires_at and current_time > post.expires_at:
                    expired_posts.append(post_id)
            
            for post_id in expired_posts:
                del self.posts[post_id]
                logger.info(f"Expired post removed: {post_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired posts: {e}")
    
    def _update_user_activity(self):
        """Update user activity metrics"""
        try:
            # Update daily active users
            cutoff_time = datetime.now() - timedelta(days=1)
            active_users = sum(1 for user in self.users.values() 
                             if user.last_active > cutoff_time)
            
            self.performance_stats['daily_active_users'] = active_users
            
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
    
    def _calculate_signal_accuracy(self):
        """Calculate overall signal accuracy"""
        try:
            # This would integrate with actual trading results
            # For now, using placeholder calculation
            total_signals = sum(user.total_signals for user in self.users.values())
            if total_signals > 0:
                total_accuracy = sum(user.signal_accuracy * user.total_signals 
                                   for user in self.users.values())
                self.performance_stats['signal_accuracy'] = total_accuracy / total_signals
            else:
                self.performance_stats['signal_accuracy'] = 0.0
                
        except Exception as e:
            logger.error(f"Error calculating signal accuracy: {e}")
    
    def _update_trending_posts(self):
        """Update trending posts cache"""
        try:
            # Calculate trending score for each post
            current_time = datetime.now()
            
            for post in self.posts.values():
                if post.is_deleted:
                    continue
                
                # Time decay factor
                hours_old = (current_time - post.created_at).total_seconds() / 3600
                time_decay = max(0.1, 1 / (1 + hours_old / 24))  # Decay over 24 hours
                
                # Engagement score
                engagement_score = (
                    post.likes * 1.0 +
                    post.comments * 2.0 +
                    post.shares * 3.0 +
                    post.views * 0.1
                )
                
                # Trending score
                trending_score = engagement_score * time_decay
                
                # Store in post metadata
                if not hasattr(post, 'trending_score'):
                    post.trending_score = trending_score
                else:
                    post.trending_score = trending_score
                    
        except Exception as e:
            logger.error(f"Error updating trending posts: {e}")
    
    def _clean_old_notifications(self):
        """Clean up old notifications"""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)
            
            for user_id, notifications in self.notifications.items():
                self.notifications[user_id] = [
                    notif for notif in notifications
                    if notif.created_at > cutoff_time and not notif.is_deleted
                ]
                
        except Exception as e:
            logger.error(f"Error cleaning old notifications: {e}")
    
    def _load_data(self):
        """Load existing data from storage"""
        try:
            import os
            
            if os.path.exists('social_trading_data.json'):
                with open('social_trading_data.json', 'r') as f:
                    data = json.load(f)
                    
                    # Load users
                    for user_data in data.get('users', []):
                        user = SocialUser(**user_data)
                        user.created_at = datetime.fromisoformat(user_data['created_at'])
                        user.updated_at = datetime.fromisoformat(user_data['updated_at'])
                        user.last_active = datetime.fromisoformat(user_data['last_active'])
                        user.following = set(user_data.get('following', []))
                        user.followers = set(user_data.get('followers', []))
                        user.blocked_users = set(user_data.get('blocked_users', []))
                        self.users[user.id] = user
                    
                    # Load posts
                    for post_data in data.get('posts', []):
                        post = SocialPost(**post_data)
                        post.created_at = datetime.fromisoformat(post_data['created_at'])
                        post.updated_at = datetime.fromisoformat(post_data['updated_at'])
                        if post_data.get('expires_at'):
                            post.expires_at = datetime.fromisoformat(post_data['expires_at'])
                        self.posts[post.id] = post
                    
                    # Load comments
                    for comment_data in data.get('comments', []):
                        comment = SocialComment(**comment_data)
                        comment.created_at = datetime.fromisoformat(comment_data['created_at'])
                        comment.updated_at = datetime.fromisoformat(comment_data['updated_at'])
                        self.comments[comment.id] = comment
                        
                logger.info("Social trading data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading social trading data: {e}")
    
    def _save_data(self):
        """Save data to storage"""
        try:
            data = {
                'users': [],
                'posts': [],
                'comments': []
            }
            
            # Save users
            for user in self.users.values():
                user_data = asdict(user)
                user_data['created_at'] = user.created_at.isoformat()
                user_data['updated_at'] = user.updated_at.isoformat()
                user_data['last_active'] = user.last_active.isoformat()
                user_data['following'] = list(user.following)
                user_data['followers'] = list(user.followers)
                user_data['blocked_users'] = list(user.blocked_users)
                data['users'].append(user_data)
            
            # Save posts
            for post in self.posts.values():
                post_data = asdict(post)
                post_data['created_at'] = post.created_at.isoformat()
                post_data['updated_at'] = post.updated_at.isoformat()
                if post.expires_at:
                    post_data['expires_at'] = post.expires_at.isoformat()
                data['posts'].append(post_data)
            
            # Save comments
            for comment in self.comments.values():
                comment_data = asdict(comment)
                comment_data['created_at'] = comment.created_at.isoformat()
                comment_data['updated_at'] = comment.updated_at.isoformat()
                data['comments'].append(comment_data)
            
            with open('social_trading_data.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving social trading data: {e}")
    
    def _broadcast_to_followers(self, user_id: str, message: Dict):
        """Broadcast message to user's followers"""
        try:
            user = self.users.get(user_id)
            if not user:
                return
            
            for follower_id in user.followers:
                self._send_real_time_update(follower_id, message)
                
        except Exception as e:
            logger.error(f"Error broadcasting to followers: {e}")
    
    def _send_real_time_update(self, user_id: str, message: Dict):
        """Send real-time update to user"""
        try:
            for callback in self.websocket_callbacks:
                callback(user_id, message)
        except Exception as e:
            logger.error(f"Error sending real-time update: {e}")
    
    def _create_notification(self, user_id: str, notification_type: NotificationType,
                           title: str, message: str, **kwargs):
        """Create a notification for a user"""
        try:
            notification = SocialNotification(
                id=str(uuid4()),
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                message=message,
                **kwargs
            )
            
            if user_id not in self.notifications:
                self.notifications[user_id] = []
            
            self.notifications[user_id].append(notification)
            
            # Send real-time notification
            self._send_real_time_update(user_id, {
                'type': 'notification',
                'notification': asdict(notification)
            })
            
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
    
    def _moderate_content(self, content: str) -> bool:
        """Moderate content for inappropriate material"""
        try:
            if not self.auto_moderation_enabled:
                return True
            
            # Check for banned words
            content_lower = content.lower()
            for word in self.banned_words:
                if word in content_lower:
                    return False
            
            # Additional moderation rules can be added here
            
            return True
            
        except Exception as e:
            logger.error(f"Error moderating content: {e}")
            return True
    
    def _calculate_reputation_score(self, user_id: str) -> int:
        """Calculate reputation score for a user"""
        try:
            user = self.users.get(user_id)
            if not user:
                return 0
            
            # Base score calculation
            score = 0
            
            # Engagement metrics
            score += user.total_likes_received * 2
            score += user.total_comments_received * 3
            score += user.followers_count * 5
            
            # Quality metrics
            if user.signal_accuracy > 0.7:
                score += 100
            elif user.signal_accuracy > 0.6:
                score += 50
            
            # Verification bonus
            if user.is_verified:
                score += 200
            
            # Role bonus
            role_bonus = {
                UserRole.BEGINNER: 0,
                UserRole.INTERMEDIATE: 25,
                UserRole.ADVANCED: 50,
                UserRole.EXPERT: 100,
                UserRole.VERIFIED: 200,
                UserRole.MODERATOR: 300
            }
            score += role_bonus.get(user.role, 0)
            
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Error calculating reputation score: {e}")
            return 0
    
    # Feed algorithms
    
    def _get_chronological_feed(self, user_id: str, limit: int = 20) -> List[SocialPost]:
        """Get chronological feed"""
        try:
            posts = [post for post in self.posts.values() 
                    if not post.is_deleted and self._can_user_see_post(user_id, post)]
            
            posts.sort(key=lambda x: x.created_at, reverse=True)
            return posts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting chronological feed: {e}")
            return []
    
    def _get_trending_feed(self, user_id: str, limit: int = 20) -> List[SocialPost]:
        """Get trending feed"""
        try:
            posts = [post for post in self.posts.values() 
                    if not post.is_deleted and self._can_user_see_post(user_id, post)]
            
            # Sort by trending score
            posts.sort(key=lambda x: getattr(x, 'trending_score', 0), reverse=True)
            return posts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending feed: {e}")
            return []
    
    def _get_personalized_feed(self, user_id: str, limit: int = 20) -> List[SocialPost]:
        """Get personalized feed based on user preferences"""
        try:
            user = self.users.get(user_id)
            if not user:
                return self._get_chronological_feed(user_id, limit)
            
            posts = [post for post in self.posts.values() 
                    if not post.is_deleted and self._can_user_see_post(user_id, post)]
            
            # Score posts based on personalization factors
            for post in posts:
                score = 0
                
                # Following bonus
                if post.author_id in user.following:
                    score += 50
                
                # Interaction history (simplified)
                score += getattr(post, 'trending_score', 0)
                
                # Store personalization score
                post.personalization_score = score
            
            posts.sort(key=lambda x: getattr(x, 'personalization_score', 0), reverse=True)
            return posts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting personalized feed: {e}")
            return self._get_chronological_feed(user_id, limit)
    
    def _get_following_feed(self, user_id: str, limit: int = 20) -> List[SocialPost]:
        """Get feed from users being followed"""
        try:
            user = self.users.get(user_id)
            if not user:
                return []
            
            posts = [post for post in self.posts.values() 
                    if not post.is_deleted and post.author_id in user.following]
            
            posts.sort(key=lambda x: x.created_at, reverse=True)
            return posts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting following feed: {e}")
            return []
    
    def _can_user_see_post(self, user_id: str, post: SocialPost) -> bool:
        """Check if user can see a post"""
        try:
            user = self.users.get(user_id)
            author = self.users.get(post.author_id)
            
            if not user or not author:
                return False
            
            # Check if user is blocked
            if user_id in author.blocked_users:
                return False
            
            # Check privacy settings
            if author.privacy_settings.get('profile_visibility') == 'private':
                if user_id not in author.followers and user_id != post.author_id:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking post visibility: {e}")
            return False
    
    # Public API methods
    
    def create_user(self, username: str, display_name: str, email: str, **kwargs) -> str:
        """Create a new social trading user"""
        try:
            user_id = str(uuid4())
            user = SocialUser(
                id=user_id,
                username=username,
                display_name=display_name,
                email=email,
                role=UserRole.BEGINNER,
                **kwargs
            )
            
            self.users[user_id] = user
            self.performance_stats['total_users'] += 1
            
            self._save_data()
            
            logger.info(f"Social user created: {user_id} ({username})")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def create_post(self, author_id: str, post_type: PostType, content: str, **kwargs) -> str:
        """Create a new social post"""
        try:
            # Validate user
            if author_id not in self.users:
                raise ValueError("Invalid author ID")
            
            # Moderate content
            if not self._moderate_content(content):
                raise ValueError("Content rejected by moderation")
            
            post_id = str(uuid4())
            post = SocialPost(
                id=post_id,
                author_id=author_id,
                author_name=self.users[author_id].display_name,
                post_type=post_type,
                content=content,
                **kwargs
            )
            
            self.posts[post_id] = post
            
            # Update user stats
            user = self.users[author_id]
            user.total_posts += 1
            if post_type == PostType.TRADE_SIGNAL:
                user.total_signals += 1
            
            # Update performance stats
            self.performance_stats['total_posts'] += 1
            
            # Process mentions
            self._process_mentions(post)
            
            # Broadcast to followers
            self._broadcast_to_followers(author_id, {
                'type': 'new_post',
                'post': asdict(post)
            })
            
            self._save_data()
            
            logger.info(f"Social post created: {post_id}")
            return post_id
            
        except Exception as e:
            logger.error(f"Error creating post: {e}")
            return None
    
    def create_comment(self, post_id: str, author_id: str, content: str, 
                      parent_comment_id: str = None) -> str:
        """Create a comment on a post"""
        try:
            # Validate post and user
            if post_id not in self.posts or author_id not in self.users:
                raise ValueError("Invalid post or author ID")
            
            # Moderate content
            if not self._moderate_content(content):
                raise ValueError("Content rejected by moderation")
            
            comment_id = str(uuid4())
            comment = SocialComment(
                id=comment_id,
                post_id=post_id,
                author_id=author_id,
                author_name=self.users[author_id].display_name,
                content=content,
                parent_comment_id=parent_comment_id
            )
            
            self.comments[comment_id] = comment
            
            # Update post comment count
            post = self.posts[post_id]
            post.comments += 1
            
            # Update parent comment if nested
            if parent_comment_id and parent_comment_id in self.comments:
                parent_comment = self.comments[parent_comment_id]
                parent_comment.replies.append(comment_id)
            
            # Update performance stats
            self.performance_stats['total_comments'] += 1
            
            # Create notification for post author
            if post.author_id != author_id:
                self._create_notification(
                    post.author_id,
                    NotificationType.POST_COMMENT,
                    "New Comment",
                    f"{self.users[author_id].display_name} commented on your post",
                    related_post_id=post_id,
                    related_user_id=author_id,
                    related_comment_id=comment_id
                )
            
            # Broadcast update
            self._broadcast_to_followers(author_id, {
                'type': 'new_comment',
                'comment': asdict(comment)
            })
            
            self._save_data()
            
            logger.info(f"Comment created: {comment_id}")
            return comment_id
            
        except Exception as e:
            logger.error(f"Error creating comment: {e}")
            return None
    
    def like_post(self, post_id: str, user_id: str) -> bool:
        """Like a post"""
        try:
            if post_id not in self.posts or user_id not in self.users:
                return False
            
            post = self.posts[post_id]
            post.likes += 1
            
            # Update performance stats
            self.performance_stats['total_likes'] += 1
            
            # Update author stats
            author = self.users[post.author_id]
            author.total_likes_received += 1
            
            # Create notification
            if post.author_id != user_id:
                self._create_notification(
                    post.author_id,
                    NotificationType.POST_LIKE,
                    "Post Liked",
                    f"{self.users[user_id].display_name} liked your post",
                    related_post_id=post_id,
                    related_user_id=user_id
                )
            
            self._save_data()
            
            logger.info(f"Post liked: {post_id} by {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error liking post: {e}")
            return False
    
    def follow_user(self, follower_id: str, following_id: str) -> bool:
        """Follow a user"""
        try:
            if follower_id not in self.users or following_id not in self.users:
                return False
            
            if follower_id == following_id:
                return False
            
            follower = self.users[follower_id]
            following = self.users[following_id]
            
            # Check if already following
            if following_id in follower.following:
                return False
            
            # Update relationships
            follower.following.add(following_id)
            follower.following_count += 1
            
            following.followers.add(follower_id)
            following.followers_count += 1
            
            # Create notification
            self._create_notification(
                following_id,
                NotificationType.NEW_FOLLOWER,
                "New Follower",
                f"{follower.display_name} started following you",
                related_user_id=follower_id
            )
            
            self._save_data()
            
            logger.info(f"User follow: {follower_id} -> {following_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error following user: {e}")
            return False
    
    def unfollow_user(self, follower_id: str, following_id: str) -> bool:
        """Unfollow a user"""
        try:
            if follower_id not in self.users or following_id not in self.users:
                return False
            
            follower = self.users[follower_id]
            following = self.users[following_id]
            
            # Check if following
            if following_id not in follower.following:
                return False
            
            # Update relationships
            follower.following.remove(following_id)
            follower.following_count -= 1
            
            following.followers.remove(follower_id)
            following.followers_count -= 1
            
            self._save_data()
            
            logger.info(f"User unfollow: {follower_id} -> {following_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unfollowing user: {e}")
            return False
    
    def get_user_feed(self, user_id: str, algorithm: str = 'personalized', 
                     limit: int = 20) -> List[Dict]:
        """Get user's social feed"""
        try:
            if user_id not in self.users:
                return []
            
            # Get posts using specified algorithm
            if algorithm in self.feed_algorithms:
                posts = self.feed_algorithms[algorithm](user_id, limit)
            else:
                posts = self._get_chronological_feed(user_id, limit)
            
            # Convert to dict format
            feed = []
            for post in posts:
                post_dict = asdict(post)
                post_dict['created_at'] = post.created_at.isoformat()
                post_dict['updated_at'] = post.updated_at.isoformat()
                if post.expires_at:
                    post_dict['expires_at'] = post.expires_at.isoformat()
                
                # Add author info
                author = self.users.get(post.author_id)
                if author:
                    post_dict['author'] = {
                        'id': author.id,
                        'username': author.username,
                        'display_name': author.display_name,
                        'avatar_url': author.avatar_url,
                        'role': author.role.value,
                        'is_verified': author.is_verified
                    }
                
                feed.append(post_dict)
            
            return feed
            
        except Exception as e:
            logger.error(f"Error getting user feed: {e}")
            return []
    
    def get_post_comments(self, post_id: str, limit: int = 50) -> List[Dict]:
        """Get comments for a post"""
        try:
            if post_id not in self.posts:
                return []
            
            # Get all comments for the post
            post_comments = [comment for comment in self.comments.values() 
                           if comment.post_id == post_id and not comment.is_deleted]
            
            # Sort by creation time
            post_comments.sort(key=lambda x: x.created_at)
            
            # Convert to dict format
            comments = []
            for comment in post_comments[:limit]:
                comment_dict = asdict(comment)
                comment_dict['created_at'] = comment.created_at.isoformat()
                comment_dict['updated_at'] = comment.updated_at.isoformat()
                
                # Add author info
                author = self.users.get(comment.author_id)
                if author:
                    comment_dict['author'] = {
                        'id': author.id,
                        'username': author.username,
                        'display_name': author.display_name,
                        'avatar_url': author.avatar_url,
                        'role': author.role.value,
                        'is_verified': author.is_verified
                    }
                
                comments.append(comment_dict)
            
            return comments
            
        except Exception as e:
            logger.error(f"Error getting post comments: {e}")
            return []
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile information"""
        try:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            
            # Calculate reputation score
            reputation_score = self._calculate_reputation_score(user_id)
            user.reputation_score = reputation_score
            
            profile = asdict(user)
            profile['created_at'] = user.created_at.isoformat()
            profile['updated_at'] = user.updated_at.isoformat()
            profile['last_active'] = user.last_active.isoformat()
            profile['following'] = list(user.following)
            profile['followers'] = list(user.followers)
            profile['blocked_users'] = list(user.blocked_users)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def get_user_notifications(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user notifications"""
        try:
            if user_id not in self.notifications:
                return []
            
            notifications = self.notifications[user_id]
            
            # Sort by creation time (newest first)
            notifications.sort(key=lambda x: x.created_at, reverse=True)
            
            # Convert to dict format
            result = []
            for notification in notifications[:limit]:
                if notification.is_deleted:
                    continue
                
                notif_dict = asdict(notification)
                notif_dict['created_at'] = notification.created_at.isoformat()
                if notification.read_at:
                    notif_dict['read_at'] = notification.read_at.isoformat()
                
                result.append(notif_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark a notification as read"""
        try:
            if user_id not in self.notifications:
                return False
            
            for notification in self.notifications[user_id]:
                if notification.id == notification_id:
                    notification.is_read = True
                    notification.read_at = datetime.now()
                    self._save_data()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            return False
    
    def get_trending_posts(self, limit: int = 20) -> List[Dict]:
        """Get trending posts"""
        try:
            return self._get_trending_feed(None, limit)
        except Exception as e:
            logger.error(f"Error getting trending posts: {e}")
            return []
    
    def search_posts(self, query: str, filters: Dict = None, limit: int = 20) -> List[Dict]:
        """Search posts"""
        try:
            query_lower = query.lower()
            results = []
            
            for post in self.posts.values():
                if post.is_deleted:
                    continue
                
                # Search in content
                if query_lower in post.content.lower():
                    results.append(post)
                    continue
                
                # Search in tags
                if any(query_lower in tag.lower() for tag in post.tags):
                    results.append(post)
                    continue
                
                # Search in symbol
                if post.symbol and query_lower in post.symbol.lower():
                    results.append(post)
                    continue
            
            # Apply filters
            if filters:
                if 'post_type' in filters:
                    results = [p for p in results if p.post_type.value == filters['post_type']]
                
                if 'author_id' in filters:
                    results = [p for p in results if p.author_id == filters['author_id']]
                
                if 'symbol' in filters:
                    results = [p for p in results if p.symbol == filters['symbol']]
            
            # Sort by relevance (for now, just by creation time)
            results.sort(key=lambda x: x.created_at, reverse=True)
            
            # Convert to dict format
            result_dicts = []
            for post in results[:limit]:
                post_dict = asdict(post)
                post_dict['created_at'] = post.created_at.isoformat()
                post_dict['updated_at'] = post.updated_at.isoformat()
                
                # Add author info
                author = self.users.get(post.author_id)
                if author:
                    post_dict['author'] = {
                        'id': author.id,
                        'username': author.username,
                        'display_name': author.display_name,
                        'avatar_url': author.avatar_url,
                        'role': author.role.value,
                        'is_verified': author.is_verified
                    }
                
                result_dicts.append(post_dict)
            
            return result_dicts
            
        except Exception as e:
            logger.error(f"Error searching posts: {e}")
            return []
    
    def _process_mentions(self, post: SocialPost):
        """Process user mentions in a post"""
        try:
            content = post.content
            mentions = []
            
            # Simple mention detection (@username)
            import re
            mention_pattern = r'@([a-zA-Z0-9_]+)'
            matches = re.findall(mention_pattern, content)
            
            for username in matches:
                # Find user by username
                for user in self.users.values():
                    if user.username == username:
                        mentions.append(user.id)
                        
                        # Create notification
                        self._create_notification(
                            user.id,
                            NotificationType.MENTION,
                            "You were mentioned",
                            f"{post.author_name} mentioned you in a post",
                            related_post_id=post.id,
                            related_user_id=post.author_id
                        )
                        break
            
            post.mentioned_users = mentions
            
        except Exception as e:
            logger.error(f"Error processing mentions: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get social trading performance statistics"""
        try:
            stats = self.performance_stats.copy()
            
            # Add real-time stats
            stats['total_users'] = len(self.users)
            stats['total_posts'] = len(self.posts)
            stats['total_comments'] = len(self.comments)
            stats['active_connections'] = sum(len(connections) for connections in self.active_connections.values())
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def add_websocket_callback(self, callback: Callable):
        """Add WebSocket callback for real-time updates"""
        self.websocket_callbacks.append(callback)
    
    def remove_websocket_callback(self, callback: Callable):
        """Remove WebSocket callback"""
        if callback in self.websocket_callbacks:
            self.websocket_callbacks.remove(callback)
    
    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp"""
        try:
            if user_id in self.users:
                self.users[user_id].last_active = datetime.now()
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
    
    def shutdown(self):
        """Shutdown the social trading system"""
        try:
            self.stop_background_tasks()
            self._save_data()
            logger.info("Social trading system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")