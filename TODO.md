This TODO list is automatically generated from the cookiecutter-python-project template.
The following tasks need to be done to get a fully working project:

* Push to your remote repository for the first time by doing `git push origin main`.
* Head to your user settings at `https://pypi.org` and setup PyPI trusted publishing.
  In order to do so, you have to head to the "Publishing" tab, scroll to the bottom
  and add a "new pending publisher". The relevant information is:
  * PyPI project name: `rubix`
  * Owner: `AstroAI-Lab`
  * Repository name: `rubix`
  * Workflow name: `pypi.yml`
  * Environment name: not required
* Enable the integration of Readthedocs with your Git hoster. In the case of Github, this means
  that you need to login at [Read the Docs](https://readthedocs.org) and click the button
  *Import a Project*.
* Enable the integration with `codecov.io` by heading to the [Codecov.io Website](https://codecov.io),
  log in (e.g. with your Github credentials) and enable integration for your repository. In order to do
  so, you need to select it from the list of repositories (potentially re-syncing with GitHub) and head
  to the Settings Tab. Within setting, get your token for this repository and put store it as a secret
  called `CODECOV_TOKEN` for GitHub Actions.
