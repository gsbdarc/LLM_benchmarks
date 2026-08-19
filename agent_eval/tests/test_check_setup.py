from importlib.metadata import PackageNotFoundError
from pathlib import Path

from packaging.requirements import Requirement

from scripts import check_setup


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_requirements_source_declares_cross_branch_direct_dependencies():
    requirements = check_setup.load_requirements(REPO_ROOT / "requirements.in")
    by_name = {requirement.name.lower(): requirement for requirement in requirements}

    assert {
        "numpy",
        "openai",
        "packaging",
        "pillow",
        "pymongo",
        "requests",
        "seaborn",
    } <= by_name.keys()
    assert "2.54.0" in by_name["openai"].specifier
    assert "3.0.0" not in by_name["openai"].specifier


def test_generated_lock_is_exact_and_satisfies_requirements_source():
    source = check_setup.load_requirements(REPO_ROOT / "requirements.in")
    lock = check_setup.load_requirements(REPO_ROOT / "requirements.txt")

    checks = check_setup.check_lock(source, lock)

    assert all(check.ok for check in checks), check_setup.render_checks(checks)


def test_only_python_310_is_supported():
    assert check_setup.check_python_version((3, 10)).ok
    assert not check_setup.check_python_version((3, 9)).ok
    assert not check_setup.check_python_version((3, 11)).ok
    assert not check_setup.check_python_version((3, 12)).ok


def test_requirement_checks_detect_missing_and_incompatible_packages():
    versions = {"openai": "3.0.0"}

    def installed_version(name: str) -> str:
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    checks = check_setup.check_requirements(
        [Requirement("openai>=2,<3"), Requirement("pymongo")],
        installed_version=installed_version,
    )

    assert [(check.name, check.ok) for check in checks] == [
        ("package openai", False),
        ("package pymongo", False),
    ]


def test_inference_profile_accepts_dotenv_without_revealing_values(tmp_path):
    secret = "do-not-print-this-value"
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"BASE_DIR={tmp_path}",
                f"STANFORD_API_KEY={secret}",
                "MONGO_DB_USERNAME=researcher",
                "MONGO_DB_PASSWORD=database-secret",
            ]
        ),
        encoding="utf-8",
    )

    checks = check_setup.check_environment(
        "inference",
        root=tmp_path,
        environ={},
        which=lambda command: f"/usr/bin/{command}",
    )
    output = check_setup.render_checks(checks)

    assert {
        "command pdftoppm",
        "environment BASE_DIR",
        "environment MONGO_DB_PASSWORD",
        "environment MONGO_DB_USERNAME",
        "environment STANFORD_API_KEY",
    } <= {check.name for check in checks}
    assert all(check.ok for check in checks)
    assert secret not in output
    assert "database-secret" not in output


def test_agent_eval_playground_reports_missing_variable_names(tmp_path):
    checks = check_setup.check_environment(
        "agent-eval",
        backend="playground",
        root=tmp_path,
        environ={},
    )
    output = check_setup.render_checks(checks)

    assert "MONGO_DB_USERNAME" in output
    assert "MONGO_DB_PASSWORD" in output
    assert "STANFORD_API_KEY" in output
    assert not all(check.ok for check in checks)


def test_qwen_backend_requires_local_model_url_but_nim_does_not(tmp_path):
    mongo = {
        "MONGO_DB_USERNAME": "configured",
        "MONGO_DB_PASSWORD": "configured",
    }

    qwen_checks = check_setup.check_environment(
        "agent-eval", backend="qwen", root=tmp_path, environ=mongo
    )
    nim_checks = check_setup.check_environment(
        "agent-eval", backend="nim", root=tmp_path, environ=mongo
    )

    assert any(
        check.name == "environment LOCAL_MODEL_URL" and not check.ok
        for check in qwen_checks
    )
    assert all(check.name != "environment LOCAL_MODEL_URL" for check in nim_checks)
    assert all(check.ok for check in nim_checks)


def test_batch_mode_requires_slurm(tmp_path):
    checks = check_setup.check_environment(
        "agent-eval",
        backend="nim",
        batch=True,
        root=tmp_path,
        environ={
            "MONGO_DB_USERNAME": "configured",
            "MONGO_DB_PASSWORD": "configured",
        },
        which=lambda _command: None,
    )

    assert any(check.name == "command sbatch" and not check.ok for check in checks)


def test_main_returns_nonzero_when_a_required_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(check_setup, "load_requirements", lambda _path: [])
    monkeypatch.setattr(
        check_setup,
        "check_python_version",
        lambda: check_setup.Check("python version", True, "supported"),
    )
    monkeypatch.setattr(
        check_setup,
        "check_dependency_graph",
        lambda: check_setup.Check("pip dependency graph", False, "broken"),
    )

    exit_code = check_setup.main(["python"])

    assert exit_code == 1
    assert "FAIL  pip dependency graph" in capsys.readouterr().out
