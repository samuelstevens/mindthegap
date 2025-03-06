"""Tests for the newt module in small_data_metrics package."""

from small_data_metrics import config, newt


def test_include_task_no_filters():
    """Test include_task with no filters (should include all tasks)."""
    cfg = config.Newt()
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True


def test_include_task_exclude_tasks():
    """Test include_task with exclude_tasks filter."""
    cfg = config.Newt(exclude_tasks=["task1", "task2"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
    assert newt.include_task(cfg, "task3", "cluster1", "subcluster1") is True


def test_include_task_exclude_clusters():
    """Test include_task with exclude_clusters filter."""
    cfg = config.Newt(exclude_clusters=["cluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
    assert newt.include_task(cfg, "task1", "cluster2", "subcluster1") is True


def test_include_task_exclude_subclusters():
    """Test include_task with exclude_subclusters filter."""
    cfg = config.Newt(exclude_subclusters=["subcluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster2") is True
    assert newt.include_task(cfg, "task1", "cluster1", None) is True


def test_include_task_specific_tasks():
    """Test include_task with specific tasks inclusion."""
    cfg = config.Newt(tasks=["task1", "task2"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    assert newt.include_task(cfg, "task3", "cluster1", "subcluster1") is False


def test_include_task_include_clusters():
    """Test include_task with include_clusters filter."""
    cfg = config.Newt(include_clusters=["cluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    assert newt.include_task(cfg, "task1", "cluster2", "subcluster1") is False


def test_include_task_include_subclusters():
    """Test include_task with include_subclusters filter."""
    cfg = config.Newt(include_subclusters=["subcluster1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster2") is False
    assert newt.include_task(cfg, "task1", "cluster1", None) is False


def test_include_task_multiple_inclusion_filters():
    """Test include_task with multiple inclusion filters."""
    cfg = config.Newt(
        tasks=["task2"],
        include_clusters=["cluster1"],
        include_subclusters=["subcluster2"],
    )
    # Matches cluster
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is True
    # Matches subcluster
    assert newt.include_task(cfg, "task1", "cluster2", "subcluster2") is True
    # Matches task
    assert newt.include_task(cfg, "task2", "cluster2", "subcluster3") is True
    # Matches none
    assert newt.include_task(cfg, "task3", "cluster3", "subcluster3") is False


def test_include_task_inclusion_exclusion_precedence():
    """Test that exclusion takes precedence over inclusion."""
    cfg = config.Newt(tasks=["task1"], exclude_tasks=["task1"])
    assert newt.include_task(cfg, "task1", "cluster1", "subcluster1") is False
