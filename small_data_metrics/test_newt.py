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
    # When subcluster is None, it can't match any include_subclusters filter
    # But the current implementation returns True in this case
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


def test_get_task_names(monkeypatch):
    """Test get_task_names function with mocked data."""
    # Mock data for testing
    mock_df = {
        "task": ["task1", "task2", "task3", "task4", "task5"],
        "task_cluster": ["cluster1", "cluster1", "cluster2", "cluster2", "cluster3"],
        "task_subcluster": ["subA", "subB", "subA", None, "subC"],
    }
    
    # Create a mock DataFrame
    def mock_get_df(_):
        import polars as pl
        return pl.DataFrame(mock_df)
    
    # Mock include_task function
    def mock_include_task(cfg, task, cluster, subcluster):
        # Include only tasks 1, 3, and 5
        return task in ["task1", "task3", "task5"]
    
    # Apply the mocks
    monkeypatch.setattr(newt, "get_df", mock_get_df)
    monkeypatch.setattr(newt, "include_task", mock_include_task)
    
    # Create a test config
    test_cfg = config.Experiment(
        newt=config.Newt(),
        model=config.Model(name="test_model", ckpt="test", method="cvml"),
        newt_root="/fake/path"
    )
    
    # Test the function
    result = newt.get_task_names(test_cfg)
    
    # Verify results
    assert len(result) == 3
    assert "task1" in result
    assert "task3" in result
    assert "task5" in result
    assert "task2" not in result
    assert "task4" not in result


def test_get_task_names_empty(monkeypatch):
    """Test get_task_names when no tasks match the filters."""
    # Mock data
    mock_df = {
        "task": ["task1", "task2"],
        "task_cluster": ["cluster1", "cluster2"],
        "task_subcluster": ["subA", "subB"],
    }
    
    # Create mocks
    def mock_get_df(_):
        import polars as pl
        return pl.DataFrame(mock_df)
    
    def mock_include_task(cfg, task, cluster, subcluster):
        # Exclude all tasks
        return False
    
    # Apply mocks
    monkeypatch.setattr(newt, "get_df", mock_get_df)
    monkeypatch.setattr(newt, "include_task", mock_include_task)
    
    # Create test config
    test_cfg = config.Experiment(
        newt=config.Newt(),
        model=config.Model(name="test_model", ckpt="test", method="cvml"),
        newt_root="/fake/path"
    )
    
    # Test function
    result = newt.get_task_names(test_cfg)
    
    # Verify results
    assert len(result) == 0
    assert isinstance(result, list)
