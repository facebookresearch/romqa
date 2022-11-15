import tqdm
import ujson as json


def make_tables(c, tables):
    # drop existing tables
    for t in tables:
        c.execute('DROP TABLE IF EXISTS {}'.format(t))

    if 'ents' in tables:
        c.execute('CREATE TABLE ents(id INTEGER PRIMARY KEY, uri TEXT, text TEXT, aliases JSON, desc TEXT, wiki_title TEXT, UNIQUE(uri))')

    if 'props' in tables:
        c.execute('CREATE TABLE props(id INTEGER PRIMARY KEY, uri TEXT, text TEXT, aliases JSON, desc TEXT, UNIQUE(uri))')

    if 'docs' in tables:
        c.execute('CREATE TABLE docs(id INTEGER PRIMARY KEY, uri TEXT, title TEXT, text TEXT, UNIQUE(uri))')

    if 'trips' in tables:
        c.execute("""
            CREATE TABLE trips(
                id INTEGER PRIMARY KEY,
                subj_id INTEGER NOT NULL,
                obj_id INTEGER NOT NULL,
                prop_id INTEGER NOT NULL,
                UNIQUE(subj_id, obj_id, prop_id),
                FOREIGN KEY (subj_id) REFERENCES ents(id),
                FOREIGN KEY (obj_id) REFERENCES ents(id),
                FOREIGN KEY (prop_id) REFERENCES props(id)
        )""")

    if 'evidence' in tables:
        c.execute("""
            CREATE TABLE evidence(
                id INTEGER PRIMARY KEY,
                trip_id INTEGER NOT NULL,
                doc_id INTEGER NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                UNIQUE(trip_id, doc_id, start, end),
                FOREIGN KEY (trip_id) REFERENCES trips(id),
                FOREIGN KEY (doc_id) REFERENCES docs(id)
        )""")


def make_annotation_tables(c, tables, drop_existing=False):
    # drop existing tables
    if drop_existing:
        for t in tables:
            c.execute('DROP TABLE IF EXISTS {}'.format(t))

    if 'hits' in tables:
        c.execute('CREATE TABLE hits(id TEXT PRIMARY KEY, status TEXT, data JSON, max_assignments INTERGER)')
    if 'assignments' in tables:
        c.execute("""
            CREATE TABLE assignments(
                id TEXT PRIMARY KEY,
                status TEXT,
                data JSON,
                hit_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                FOREIGN KEY (hit_id) REFERENCES hits(id)
            )
        """)
    if 'samples' in tables:
        c.execute("""
            CREATE TABLE samples(
                example_id TEXT,
                grade TEXT,
                data JSON,
                worker_id TEXT NOT NULL,
                FOREIGN KEY (worker_id) REFERENCES workers(id),
                PRIMARY KEY (example_id, worker_id)
            )
        """)


def batch_insert(db, table, items, batch_size=100000):
    items = items.copy()

    # find number of columns
    r = db.execute("PRAGMA table_info({})".format(table))
    cols = r.fetchall()
    json_cols_idx = {i for i, name, dtype, _, _, _ in cols if dtype == 'JSON'}

    if json_cols_idx:
        for i, r in enumerate(items):
            items[i] = [json.dumps(c) if ci in json_cols_idx else c for ci, c in enumerate(r)]

    for i in tqdm.trange(0, len(items), batch_size, desc='insert {}'.format(table)):
        batch = items[i:i+batch_size]
        c = db.cursor()
        c.execute('BEGIN')
        try:
            c.executemany('INSERT INTO {} VALUES({})'.format(table, ', '.join(['?'] * len(cols))), batch)
            db.commit()
        except db.Error as e:
            db.rollback()
            raise e
        finally:
            c.close()
