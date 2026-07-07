import sys, sqlite3, pathlib, os
import config
import traceback

def main():
    print('=== DATABASE DIAGNOSTIC ===')
    print('DB_PATH from config:', config.DB_PATH)
    db_path = pathlib.Path(config.DB_PATH)
    print('Absolute DB path:', db_path.resolve())
    print('DB file exists:', db_path.exists())

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
        tables = cursor.fetchall()
        print('Tables in DB:')
        if tables:
            for t in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
                count = cursor.fetchone()[0]
                print(f'  {t[0]} (rows={count})')
        else:
            print('  (NO TABLES FOUND - empty database)')
        
        # Try to see if creating table works
        try:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Users (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('Student', 'Educator', 'Admin')),
                login_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            ''')
            print('Created Users table manually without errors.')
        except Exception as e:
            print('Error creating Users table manually:', e)
            traceback.print_exc()
        
        conn.close()
    else:
        print('DB does not exist.')

if __name__ == '__main__':
    main()
