import os

import git


class Git:
    def __init__(self):
        os.chdir("../")
        self.repo = git.Repo()

    def push(self, comment):
        try:
            self.repo.git.add(".")
            self.repo.git.commit("-m", f"{comment}")
            origin = self.repo.remote(name="origin")
            origin.push()
        except:
            pass
