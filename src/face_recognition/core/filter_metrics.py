"""Metrics tracking for filter performance and validation."""

import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict, deque


class FilterMetrics:
    """Structured metrics for filter performance tracking."""

    def __init__(self):
        """Initialize metrics counters."""
        self.counters = {
            # Overall
            'total_tracks': 0,
            'sent_to_dashboard': 0,
            'filtered_out': 0,

            # Per-stage filtering
            'filtered_quality': 0,
            'filtered_confidence': 0,
            'filtered_dedup': 0,
            'filtered_rate_limit': 0,
            'filtered_track_lifetime': 0,

            # Quality breakdown
            'rejected_profile': 0,
            'rejected_blurry': 0,
            'rejected_small': 0,
            'rejected_lighting': 0,
            'rejected_occluded': 0,
            'rejected_aspect': 0,

            # Confidence breakdown
            'uncertain_matches': 0,
            'low_margin': 0,

            # Dedup stats
            'dedup_same_camera': 0,
            'dedup_cross_camera': 0,
            'dedup_upgraded': 0,
        }

        # Distribution tracking (bounded to prevent unbounded memory growth)
        self.quality_scores_sent = deque(maxlen=10000)
        self.quality_scores_filtered = deque(maxlen=10000)
        self.similarity_scores = deque(maxlen=10000)
        self.track_lifetimes = deque(maxlen=10000)

        # Detailed reason counts
        self.reason_counts = defaultdict(int)

    def log_decision(
        self,
        decision: str,
        reason: str,
        quality: float,
        similarity: float,
        lifetime: float
    ) -> None:
        """Log each filter decision.

        Args:
            decision: 'SEND' or 'FILTER'
            reason: Reason string for the decision
            quality: Quality score
            similarity: Similarity score
            lifetime: Track lifetime in seconds
        """
        self.counters['total_tracks'] += 1

        if decision == 'SEND':
            self.counters['sent_to_dashboard'] += 1
            self.quality_scores_sent.append(quality)
        else:
            self.counters['filtered_out'] += 1
            self.quality_scores_filtered.append(quality)
            self.reason_counts[reason] += 1

            # Categorize rejection reason
            if 'profile' in reason:
                self.counters['rejected_profile'] += 1
                self.counters['filtered_quality'] += 1
            elif 'blurry' in reason:
                self.counters['rejected_blurry'] += 1
                self.counters['filtered_quality'] += 1
            elif 'small' in reason:
                self.counters['rejected_small'] += 1
                self.counters['filtered_quality'] += 1
            elif 'lighting' in reason:
                self.counters['rejected_lighting'] += 1
                self.counters['filtered_quality'] += 1
            elif 'occluded' in reason:
                self.counters['rejected_occluded'] += 1
                self.counters['filtered_quality'] += 1
            elif 'aspect' in reason:
                self.counters['rejected_aspect'] += 1
                self.counters['filtered_quality'] += 1
            elif 'duplicate' in reason:
                self.counters['filtered_dedup'] += 1
                if 'same_camera' in reason:
                    self.counters['dedup_same_camera'] += 1
                elif 'cross_camera' in reason:
                    self.counters['dedup_cross_camera'] += 1
                elif 'upgraded' in reason:
                    self.counters['dedup_upgraded'] += 1
            elif 'uncertain' in reason or 'similarity_too_high' in reason:
                self.counters['filtered_confidence'] += 1
                self.counters['uncertain_matches'] += 1
            elif 'short_track' in reason or 'track_too_short' in reason:
                self.counters['filtered_track_lifetime'] += 1
            elif 'rate_limited' in reason:
                self.counters['filtered_rate_limit'] += 1

        self.similarity_scores.append(similarity)
        self.track_lifetimes.append(lifetime)

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        total = self.counters['total_tracks']
        if total == 0:
            return {
                'total_tracks': 0,
                'reduction_rate': '0.0%',
            }

        return {
            # Overall metrics
            'total_tracks': total,
            'sent_count': self.counters['sent_to_dashboard'],
            'filtered_count': self.counters['filtered_out'],
            'reduction_rate': f"{100 * self.counters['filtered_out'] / total:.1f}%",

            # Quality metrics
            'avg_quality_sent': round(np.mean(self.quality_scores_sent), 3) if self.quality_scores_sent else 0.0,
            'avg_quality_filtered': round(np.mean(self.quality_scores_filtered), 3) if self.quality_scores_filtered else 0.0,
            'avg_similarity': round(np.mean(self.similarity_scores), 3) if self.similarity_scores else 0.0,
            'avg_lifetime': round(np.mean(self.track_lifetimes), 2) if self.track_lifetimes else 0.0,

            # Filter breakdown
            'filter_breakdown': {
                'quality': self.counters['filtered_quality'],
                'confidence': self.counters['filtered_confidence'],
                'dedup': self.counters['filtered_dedup'],
                'track_lifetime': self.counters['filtered_track_lifetime'],
                'rate_limit': self.counters['filtered_rate_limit'],
            },

            # Quality rejections detail
            'quality_rejections': {
                'profile': self.counters['rejected_profile'],
                'blurry': self.counters['rejected_blurry'],
                'small': self.counters['rejected_small'],
                'lighting': self.counters['rejected_lighting'],
                'occluded': self.counters['rejected_occluded'],
                'aspect': self.counters['rejected_aspect'],
            },

            # Dedup stats
            'dedup_stats': {
                'same_camera': self.counters['dedup_same_camera'],
                'cross_camera': self.counters['dedup_cross_camera'],
                'upgraded': self.counters['dedup_upgraded'],
            },

            # Top rejection reasons
            'top_reasons': dict(sorted(
                self.reason_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }

    def reset(self) -> None:
        """Reset all counters and distributions."""
        self.counters = {k: 0 for k in self.counters}
        self.quality_scores_sent.clear()
        self.quality_scores_filtered.clear()
        self.similarity_scores.clear()
        self.track_lifetimes.clear()
        self.reason_counts.clear()

    def to_json(self) -> str:
        """Convert summary to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.get_summary(), indent=2)
