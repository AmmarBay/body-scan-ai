# User submissions

After a user agrees to data storage and a body scan completes successfully,
the app stores the submitted measurements and screening result in
`bodyfit_submissions.sqlite3` in this directory. The captured photo is not
stored. The database is local to the app server and is ignored by Git.

To use a different database location, set `BODYFIT_DB_PATH` in the app
server's environment. For hosted deployments, use a persistent disk or a
managed database: files on an ephemeral server filesystem may be lost when the
instance is replaced. The database contains sensitive body measurements;
restrict access to the server and its backups.
