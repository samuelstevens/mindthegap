"""Tests for the newt module in small_data_metrics package."""

from small_data_metrics import config, newt


def test_include_task():
    """Test the include_task function with various configurations."""
    # Test with no filters (should include all tasks)
    cfg = config.Newt()
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True

    # Test with exclude_tasks
    cfg = config.Newt(exclude_tasks=["task1", "task2"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
    assert newt.include_task(cfg, "task3", "cluster1", "subcluster1") is True

    # Test with exclude_clusters
    cfg = config.Newt(exclude_clusters=["cluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
    assert newt.include_task(cfg, "task1", "cluster2", "subcluster1") is True

    # Test with exclude_subclusters
    cfg = config.Newt(exclude_subclusters=["subcluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster2") is True
    assert newt.include_task(cfg, "task1", "cluster1", None) is True

    # Test with specific tasks inclusion
    cfg = config.Newt(tasks=["task1", "task2"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    assert newt.include_task(cfg, "task3", "cluster1", "subcluster1") is False

    # Test with include_clusters
    cfg = config.Newt(include_clusters=["cluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    assert newt.include_task(cfg, "task1", "cluster2", "subcluster1") is False

    # Test with include_subclusters
    cfg = config.Newt(include_subclusters=["subcluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster2") is False
    assert newt.include_task(cfg, "task1", "cluster1", None) is False

    # Test with multiple inclusion filters (task should match at least one)
    cfg = config.Newt(
        tasks=["task2"],
        include_clusters=["cluster1"],
        include_subclusters=["subcluster2"],
    )
    assert (
        newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    )  # Matches cluster
    assert (
        newt.include_task(cfg, "task1", "cluster2", "subcluster2") is True
    )  # Matches subcluster
    assert (
        newt.include_task(cfg, "task2", "cluster2", "subcluster3") is True
    )  # Matches task
    assert (
        newt.include_task(cfg, "task3", "cluster3", "subcluster3") is False
    )  # Matches none

    # Test with both inclusion and exclusion (exclusion takes precedence)
    cfg = config.Newt(tasks=["task1"], exclude_tasks=["task1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
