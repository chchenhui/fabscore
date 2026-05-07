# -*- coding: utf-8 -*-
"""
Docker Executor Module
Used to run YuLan-OneSim simulation scenarios in a Docker container and capture output.
"""

import subprocess
import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
from pathlib import Path
import json
import tempfile
import os


class DockerRunResult:
    """Encapsulation of Docker run results."""
    
    def __init__(self, 
                 return_code: int,
                 stdout: str = "",
                 stderr: str = "",
                 execution_time: float = 0.0,
                 timeout: bool = False,
                 error: Optional[Exception] = None):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time
        self.timeout = timeout
        self.error = error
        
    @property
    def success(self) -> bool:
        """Check if the run was successful."""
        # Check for timeout or exception first
        if self.timeout or self.error is not None:
            return False
        
        # For graceful shutdown scenarios, also check stderr for actual errors
        # even when return_code is 0
        if self.stderr:
            # Check for common error patterns in stderr
            error_indicators = [
                'error:', 'Error:', 'ERROR:',
                'exception:', 'Exception:', 'EXCEPTION:',
                'traceback', 'Traceback', 'TRACEBACK',
                'failed:', 'Failed:', 'FAILED:',
                'fatal:', 'Fatal:', 'FATAL:',
                'ImportError', 'ModuleNotFoundError', 'AttributeError',
                'NameError', 'TypeError', 'ValueError', 'KeyError'
            ]
            
            stderr_lower = self.stderr.lower()
            for indicator in error_indicators:
                if indicator.lower() in stderr_lower:
                    return False
        
        # If return_code is 0 and no error indicators in stderr, consider it successful
        return self.return_code == 0
        
    @property
    def has_errors(self) -> bool:
        """Check if there is any error output."""
        return bool(self.stderr) or self.return_code != 0 or self.error is not None
        
    def has_stderr_errors(self) -> bool:
        """
        Check if stderr contains actual error content (not just warnings or info).
        Useful for graceful shutdown scenarios where return_code is 0 but errors exist.
        """
        if not self.stderr:
            return False
            
        # Check for actual error patterns (more strict than success property)
        critical_error_indicators = [
            'error:', 'Error:', 'ERROR:',
            'exception:', 'Exception:', 'EXCEPTION:', 
            'traceback', 'Traceback', 'TRACEBACK',
            'ImportError', 'ModuleNotFoundError', 'AttributeError',
            'NameError', 'TypeError', 'ValueError', 'KeyError',
            'SyntaxError', 'IndentationError', 'RuntimeError'
        ]
        
        stderr_lower = self.stderr.lower()
        for indicator in critical_error_indicators:
            if indicator.lower() in stderr_lower:
                return True
                
        return False
    
    def get_error_summary(self) -> str:
        """Get a summary of the error for logging purposes."""
        if self.error:
            return f"Exception: {str(self.error)}"
        elif self.timeout:
            return "Execution timed out"
        elif self.return_code != 0:
            return f"Non-zero exit code: {self.return_code}"
        elif self.has_stderr_errors():
            return f"Errors in stderr (graceful shutdown): {self.stderr[:100]}..."
        elif self.stderr:
            return f"Stderr content (warnings/info): {self.stderr[:100]}..."
        else:
            return "No errors detected"


class DockerRunner:
    """Docker executor for running YuLan-OneSim in a Docker container."""
    
    def __init__(self, 
                 container_name: str = "yulan-onesim",
                 host_project_path: Optional[str] = None,
                 container_workdir: str = "/app",
                 image_name: str = "ptss/yulan-onesim-backend:latest",
                 default_timeout: int = 300):
        """
        Initializes the Docker executor.
        
        Args:
            container_name: Docker container name.
            host_project_path: Project path on the host machine. Defaults to current working directory.
            container_workdir: Project path inside the container.
            image_name: Docker image to use for running the container.
            default_timeout: Default timeout in seconds.
        """
        self.container_name = container_name
        self.host_project_path = host_project_path or os.path.abspath(os.getcwd())
        self.container_workdir = container_workdir
        self.image_name = image_name
        self.default_timeout = default_timeout
    
    def _get_current_user_info(self) -> Tuple[int, int, str]:
        """
        Get current user information for Docker operations.
        
        Returns:
            Tuple[int, int, str]: (uid, gid, username)
        """
        import os
        import pwd
        current_uid = os.getuid()
        current_gid = os.getgid()
        try:
            current_user = pwd.getpwuid(current_uid).pw_name
        except KeyError:
            current_user = str(current_uid)
        return current_uid, current_gid, current_user
        
    async def check_container_status(self) -> bool:
        """Check if the Docker container is running."""
        try:
            result = await self._run_command([
                "docker", "ps", "--filter", f"name={self.container_name}", 
                "--format", "{{.Names}}"
            ], timeout=10)
            
            return self.container_name in result.stdout
            
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False

    async def prepare_container(self, env_name: str) -> bool:
        """
        Prepares the container for a debugging session.
        It stops and removes any existing container with the same name,
        then starts a new one with the specified environment mounted.

        Args:
            env_name: The name of the environment to mount.
        
        Returns:
            bool: True if the container was prepared successfully, False otherwise.
        """
        logger.info(f"Preparing container '{self.container_name}' for environment '{env_name}'...")

        # Stop and remove existing container to ensure a clean state
        await self.cleanup_container()

        host_env_path = Path(self.host_project_path) / "src" / "envs" / env_name
        if not host_env_path.exists():
            logger.error(f"Environment path on host does not exist: {host_env_path}")
            return False
        
        # Ensure log directories exist with proper permissions
        log_dirs = [
            host_env_path / "log",
            host_env_path / "log" / "run",
            host_env_path / "log" / "debug_history",
            host_env_path / "log" / "code_backups"
        ]
        
        for log_dir in log_dirs:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                # Set permissions to be writable by the user
                log_dir.chmod(0o755)
            except Exception as e:
                logger.warning(f"Failed to create/set permissions for {log_dir}: {e}")

        container_env_path = Path(self.container_workdir) / "src" / "envs" / env_name
        
        # Mount the entire project to ensure all files are accessible
        project_volume_mount = f"{self.host_project_path}:{self.container_workdir}"

        # Get current user info for proper file permissions
        current_uid, current_gid, current_user = self._get_current_user_info()
        
        # Command to run a new container with the environment mounted
        # We use a command that keeps the container alive, like 'tail -f /dev/null'.
        # The Dockerfile's CMD will be overridden.
        # Create user inside container to avoid "I have no name!" and ensure proper file permissions
        run_cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-v", project_volume_mount,  # Mount entire project
            "-w", self.container_workdir,
            self.image_name,
            "sh", "-c", 
            f"(groupadd -g {current_gid} {current_user} 2>/dev/null || true) && "
            f"(useradd -u {current_uid} -g {current_gid} -m -s /bin/bash {current_user} 2>/dev/null || true) && "
            f"su {current_user} -c 'tail -f /dev/null'"
        ]

        # Alternative simpler approach (with "I have no name!" issue but correct file permissions):
        # run_cmd = [
        #     "docker", "run", "-d",
        #     "--name", self.container_name,
        #     "--user", f"{current_uid}:{current_gid}",  # Run as current user (shows "I have no name!")
        #     "-v", project_volume_mount,  # Mount entire project
        #     "-w", self.container_workdir,
        #     self.image_name,
        #     "tail", "-f", "/dev/null"  # Keep container running
        # ]

        # Current approach: Create user inside container to avoid "I have no name!" and ensure proper file permissions

        logger.info(f"Running new container with command: {' '.join(run_cmd)}")
        result = await self._run_command(run_cmd, timeout=60)

        if not result.success:
            logger.error(f"Failed to start new container: {result.stderr}")
            return False

        # Wait a moment for the container to be fully up
        await asyncio.sleep(5)
        
        is_running = await self.check_container_status()
        if is_running:
            logger.info(f"Container '{self.container_name}' started successfully with env '{env_name}' mounted.")
        else:
            logger.error(f"Container '{self.container_name}' failed to start.")
        
        return is_running

    async def cleanup_container(self):
        """Stops and removes the container."""
        logger.info(f"Cleaning up container '{self.container_name}'...")

        # Stop the container
        stop_result = await self._run_command(["docker", "stop", self.container_name], timeout=30)
        if stop_result.return_code == 0:
            logger.info(f"Container '{self.container_name}' stopped.")
        # We ignore errors here, as the container might not exist.

        # Remove the container
        rm_result = await self._run_command(["docker", "rm", self.container_name], timeout=30)
        if rm_result.return_code == 0:
            logger.info(f"Container '{self.container_name}' removed.")
        # We ignore errors here as well.
    
    async def start_container_if_needed(self) -> bool:
        """If the container is not running, start it."""
        if await self.check_container_status():
            logger.info(f"Container {self.container_name} is already running.")
            return True
            
        try:
            # Try to start the container
            logger.info(f"Starting container {self.container_name}...")
            result = await self._run_command([
                "docker", "start", self.container_name
            ], timeout=30)
            
            if result.success:
                # Wait for the container to fully start
                await asyncio.sleep(5)
                return await self.check_container_status()
            else:
                logger.error(f"Failed to start container: {result.stderr}. It may not exist. Please run `prepare_container` first.")
                return False
                
        except Exception as e:
            logger.error(f"Exception while starting container: {e}")
            return False
    
    async def run_simulation(self, 
                           env_name: str,
                           config_path: str = "config/config.json",
                           model_config_path: str = "config/model_config.json",
                           mode: str = "single",
                           max_steps: Optional[int] = None,
                           timeout: Optional[int] = None) -> DockerRunResult:
        """
        Runs a simulation scenario in the Docker container.
        
        Args:
            env_name: Environment name.
            config_path: Path to the configuration file.
            model_config_path: Path to the model configuration file.
            mode: Run mode.
            max_steps: Maximum number of steps.
            timeout: Timeout in seconds.
            
        Returns:
            DockerRunResult: The result of the run.
        """
        # Ensure the container is running
        if not await self.start_container_if_needed():
            return DockerRunResult(
                return_code=-1,
                error=RuntimeError("Docker container failed to start.")
            )
        
        # Build the command
        cmd_parts = [
            "python","src/main.py",
            "--config", config_path,
            "--model_config", model_config_path,
            "--mode", mode,
            "--env", env_name
        ]
        
        if max_steps is not None:
            # Modifying steps might require temporary config file changes
            pass
        
        # Get current user to maintain consistency
        _, _, current_user = self._get_current_user_info()
            
        docker_cmd = [
            "docker", "exec", "-u", current_user, "-w", self.container_workdir,
            self.container_name
        ] + cmd_parts
        
        logger.info(f"Executing Docker command: {' '.join(docker_cmd)}")
        
        return await self._run_command(
            docker_cmd, 
            timeout=timeout or self.default_timeout
        )
    
    async def run_python_script(self, 
                               script_path: str,
                               args: Optional[List[str]] = None,
                               timeout: Optional[int] = None) -> DockerRunResult:
        """
        Runs a Python script in the Docker container.
        
        Args:
            script_path: Path to the script (relative to the project root).
            args: Script arguments.
            timeout: Timeout in seconds.
            
        Returns:
            DockerRunResult: The result of the run.
        """
        if not await self.start_container_if_needed():
            return DockerRunResult(
                return_code=-1,
                error=RuntimeError("Docker container failed to start.")
            )
        
        cmd_parts = ["python", script_path]
        if args:
            cmd_parts.extend(args)
        
        # Get current user to maintain consistency
        _, _, current_user = self._get_current_user_info()
            
        docker_cmd = [
            "docker", "exec", "-u", current_user, "-w", self.container_workdir,
            self.container_name
        ] + cmd_parts
        
        logger.info(f"Executing Python script: {' '.join(docker_cmd)}")
        
        return await self._run_command(
            docker_cmd,
            timeout=timeout or self.default_timeout
        )
    
    async def copy_file_to_container(self, 
                                   local_path: str, 
                                   container_path: str) -> bool:
        """
        Copies a file to the Docker container.
        
        Args:
            local_path: Path to the local file.
            container_path: Path to the file in the container.
            
        Returns:
            bool: True on success, False otherwise.
        """
        try:
            cmd = [
                "docker", "cp", local_path,
                f"{self.container_name}:{container_path}"
            ]
            
            result = await self._run_command(cmd, timeout=30)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to copy file to container: {e}")
            return False
    
    async def copy_file_from_container(self, 
                                     container_path: str,
                                     local_path: str) -> bool:
        """
        Copies a file from the Docker container.
        
        Args:
            container_path: Path to the file in the container.
            local_path: Path to the local file.
            
        Returns:
            bool: True on success, False otherwise.
        """
        try:
            cmd = [
                "docker", "cp",
                f"{self.container_name}:{container_path}",
                local_path
            ]
            
            result = await self._run_command(cmd, timeout=30)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to copy file from container: {e}")
            return False
    
    async def execute_command(self, 
                            command: str,
                            timeout: Optional[int] = None) -> DockerRunResult:
        """
        Executes an arbitrary command in the Docker container.
        
        Args:
            command: The command to execute.
            timeout: Timeout in seconds.
            
        Returns:
            DockerRunResult: The result of the run.
        """
        if not await self.start_container_if_needed():
            return DockerRunResult(
                return_code=-1,
                error=RuntimeError("Docker container failed to start.")
            )
        
        # Get current user to maintain consistency
        _, _, current_user = self._get_current_user_info()
        
        docker_cmd = [
            "docker", "exec", "-u", current_user, "-w", self.container_workdir,
            self.container_name, "bash", "-c", command
        ]
        
        return await self._run_command(
            docker_cmd,
            timeout=timeout or self.default_timeout
        )
    
    async def _run_command(self, 
                          cmd: List[str], 
                          timeout: int = 600) -> DockerRunResult:
        """
        Asynchronously executes a command and captures its output.
        
        Args:
            cmd: List of command arguments.
            timeout: Timeout in seconds.
            
        Returns:
            DockerRunResult: The result of the run.
        """
        start_time = time.time()
        process = None
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=None
            )
            
            # Wait for the process to complete or timeout
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                return DockerRunResult(
                    return_code=process.returncode,
                    stdout=stdout_data.decode('utf-8', errors='replace'),
                    stderr=stderr_data.decode('utf-8', errors='replace'),
                    execution_time=execution_time,
                    timeout=False
                )
                
            except asyncio.TimeoutError:
                # Timeout handling
                if process:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                
                execution_time = time.time() - start_time
                
                return DockerRunResult(
                    return_code=-1,
                    stderr="Command execution timed out",
                    execution_time=execution_time,
                    timeout=True
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            return DockerRunResult(
                return_code=-1,
                stderr=str(e),
                execution_time=execution_time,
                error=e
            )


# Convenience functions
async def quick_run_simulation(env_name: str, 
                             timeout: int = 300,
                             container_name: str = "yulan-onesim") -> DockerRunResult:
    """
    Convenience function to quickly run a simulation scenario.
    
    Args:
        env_name: Environment name.
        timeout: Timeout in seconds.
        container_name: Container name.
        
    Returns:
        DockerRunResult: The result of the run.
    """
    # This function may need adjustment to handle the new prepare/cleanup flow.
    # For now, it assumes a container is already prepared and running.
    runner = DockerRunner(container_name=container_name)
    return await runner.run_simulation(env_name, timeout=timeout)


async def test_docker_connection(container_name: str = "yulan-onesim") -> bool:
    """
    Convenience function to test the Docker connection.
    
    Args:
        container_name: Container name.
        
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    runner = DockerRunner(container_name=container_name)
    return await runner.check_container_status()


def fix_permissions(project_path: str):
    """
    Fix file permissions for log directories to avoid Docker permission issues.
    
    Args:
        project_path: Path to the project root
    """
    import os
    import stat
    from pathlib import Path
    
    project_root = Path(project_path)
    
    # Find all log directories
    log_patterns = [
        "**/log",
        "**/log/**"
    ]
    
    for pattern in log_patterns:
        for path in project_root.glob(pattern):
            if path.is_dir():
                try:
                    # Set directory permissions to 755 (rwxr-xr-x)
                    path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                    logger.info(f"Fixed permissions for directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to fix permissions for {path}: {e}")
            elif path.is_file():
                try:
                    # Set file permissions to 644 (rw-r--r--)
                    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                    logger.info(f"Fixed permissions for file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to fix permissions for {path}: {e}")


def cleanup_docker_files(project_path: str, env_name: str = None):
    """
    Clean up Docker-created files that might have permission issues.
    
    Args:
        project_path: Path to the project root
        env_name: Specific environment name, or None for all environments
    """
    from pathlib import Path
    import subprocess
    import os
    
    project_root = Path(project_path)
    
    if env_name:
        env_paths = [project_root / "src" / "envs" / env_name]
    else:
        env_paths = list((project_root / "src" / "envs").glob("*"))
    
    for env_path in env_paths:
        if env_path.is_dir():
            log_path = env_path / "log"
            if log_path.exists():
                try:
                    # Use sudo to remove files if necessary
                    subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(log_path)], 
                                   check=False, capture_output=True)
                    logger.info(f"Fixed ownership for: {log_path}")
                except Exception as e:
                    logger.warning(f"Failed to fix ownership for {log_path}: {e}")