================
Development Help
================

Adding a remote github repository
---------------------------------

Since github removed password authentication it is best to just set up permanent access with the
personal access token like this:

.. code-block:: console

    git remote add origin https:://[username]:[access token]@github.com/[username]/[repo].git

Releasing new version
---------------------

This is a summary of the steps required to release a new version

.. code-block:: console

    poetry lock
    poetry version [ major | minor | patch ]
    poetry build
    poetry publish --username='...' --password='...'
    git commit -a -m "..."
    git push origin master
