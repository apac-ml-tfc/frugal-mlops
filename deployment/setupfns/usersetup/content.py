"""Functions to copy content into a user's home EFS directory"""

# Python Built-Ins:
import logging
import os

# External Dependencies:
from git import Repo

logger = logging.getLogger("content")


def chown_recursive(path, uid=-1, gid=-1):
    """Workaround for os.chown() not having a recursive option for folders"""
    for dirpath, dirnames, filenames in os.walk(path):
        os.chown(dirpath, uid, gid)
        for filename in filenames:
            os.chown(os.path.join(dirpath, filename), uid, gid)


def ensure_home_dir(uid):
    """Ensure a SMStudio's user ID is set up with home folder in the EFS filesystem"""
    # The root of the EFS contains folders named for each user UID, but these may not be created before
    # the user has first logged in (could os.listdir("/mnt/efs") to check):
    logger.info(f"Checking/creating home folder for user {uid}")
    home_folder = f"/mnt/efs/{uid}"
    os.makedirs(home_folder, exist_ok=True)
    # Set correct ownership permissions for this folder straight away, in case a later process errors out
    os.chown(home_folder, int(uid), -1)
    return home_folder


def clone_git_repository(efs_uid, git_repo):
    """Clone a Git repository into a SMStudio user's EFS folder (ensures the home folder exists)"""
    home_folder = ensure_home_dir(efs_uid)

    # Now ready to clone in Git content (or whatever else...)
    logger.info(f"Cloning code from repo {git_repo} to EFS user {efs_uid}")
    # Our target folder for Repo.clone_from() needs to be the *actual* target folder, not the parent
    # under which a new folder will be created, so we'll infer that from the repo name:
    repo_folder_name = git_repo.rpartition("/")[2]
    if repo_folder_name.lower().endswith(".git"):
        repo_folder_name = repo_folder_name[:-len(".git")]
    Repo.clone_from(git_repo, f"{home_folder}/{repo_folder_name}")

    # Remember to set ownership/permissions for all the stuff we just created, to give the user write
    # access:
    chown_recursive(f"{home_folder}/{repo_folder_name}", uid=int(efs_uid))
