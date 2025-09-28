const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Путь к файлу базы данных
const dbPath = path.join(__dirname, 'database', 'forum.db');

// Создаем соединение с базой данных
const db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
        console.error('Ошибка подключения к базе данных:', err.message);
    } else {
        console.log('Успешное подключение к базе данных SQLite');
        initializeDatabase();
    }
});

// Promisify обертки для всех методов DB
function dbRun(sql, params = []) {
    return new Promise((resolve, reject) => {
        db.run(sql, params, function(err) {
            if (err) reject(err);
            else resolve(this);
        });
    });
}

function dbGet(sql, params = []) {
    return new Promise((resolve, reject) => {
        db.get(sql, params, (err, row) => {
            if (err) reject(err);
            else resolve(row);
        });
    });
}

// Инициализация таблиц
async function initializeDatabase() {
    try {
        console.log('Starting database initialization...');

        // Создаем таблицы параллельно (те, что не зависят друг от друга)
        await Promise.all([
            dbRun(`CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`),
            
            dbRun(`CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                position INTEGER DEFAULT 0
            )`)
        ]);
        console.log('Users and categories tables ready');

        // Таблицы с зависимостями последовательно
        await dbRun(`CREATE TABLE IF NOT EXISTS subcategories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            position INTEGER DEFAULT 0,
            FOREIGN KEY (category_id) REFERENCES categories (id)
        )`);
        console.log('Subcategories table ready');

        await dbRun(`CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subcategory_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_pinned BOOLEAN DEFAULT 0,
            is_locked BOOLEAN DEFAULT 0,
            view_count INTEGER DEFAULT 0,
            FOREIGN KEY (subcategory_id) REFERENCES subcategories (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )`);
        console.log('Topics table ready');

        await dbRun(`CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_edited BOOLEAN DEFAULT 0,
            FOREIGN KEY (topic_id) REFERENCES topics (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )`);
        console.log('Posts table ready');

        console.log('All tables created successfully');
        
        // Добавляем тестовые данные
        await addTestUser();
        // await addTestData();
        console.log('Test data added successfully');
        
    } catch (error) {
        console.error('Database initialization error:', error);
    }
}

// Асинхронная версия addTestUser
async function addTestUser() {
    try {
        const bcrypt = require('bcryptjs');
        const testPassword = bcrypt.hashSync('password123', 10);
        
        await dbRun(
            `INSERT OR IGNORE INTO users (login, password) VALUES (?, ?)`,
            ['testuser', testPassword]
        );
        console.log('Test user created: testuser / password123');
    } catch (error) {
        console.error('Error adding test user:', error);
    }
}

// Асинхронная версия addTestData
async function addTestData() {
    try {
        // Добавляем категории
        const categories = [
            { name: 'Programming', description: 'Discussion about programming' },
            { name: 'Technology', description: 'Latest technology news' },
            { name: 'Gaming', description: 'Video games discussion' }
        ];
        
        for (const cat of categories) {
            await dbRun(
                `INSERT OR IGNORE INTO categories (name, description, position) VALUES (?, ?, ?)`,
                [cat.name, cat.description, categories.indexOf(cat) + 1]
            );
        }
        console.log('Categories added');

        // Ждем немного чтобы категории точно создались
        await new Promise(resolve => setTimeout(resolve, 100));

        // Добавляем подкатегории
        const subcategories = [
            { category: 'Programming', name: 'JavaScript', description: 'JS frameworks and libraries' },
            { category: 'Programming', name: 'Python', description: 'Python development' },
            { category: 'Technology', name: 'Hardware', description: 'PC components and gadgets' },
            { category: 'Gaming', name: 'PC Gaming', description: 'PC games discussion' }
        ];
        
        for (const subcat of subcategories) {
            const category = await dbGet('SELECT id FROM categories WHERE name = ?', [subcat.category]);
            if (category) {
                await dbRun(
                    `INSERT OR IGNORE INTO subcategories (category_id, name, description, position) VALUES (?, ?, ?, ?)`,
                    [category.id, subcat.name, subcat.description, subcategories.indexOf(subcat) + 1]
                );
            }
        }
        console.log('Subcategories added');

        // Добавляем тестовую тему и сообщения
        const user = await dbGet(`SELECT id FROM users WHERE login = 'testuser'`);
        const subcategory = await dbGet(`SELECT id FROM subcategories WHERE name = 'JavaScript'`);
        
        if (user && subcategory) {
            const result = await dbRun(
                `INSERT OR IGNORE INTO topics (subcategory_id, user_id, title, content) VALUES (?, ?, ?, ?)`,
                [subcategory.id, user.id, 'Help with Node.js async/await', 'I need help understanding async/await in Node.js...']
            );
            
            if (result.lastID) {
                const posts = [
                    'I think async/await is much better than callbacks!',
                    'You should try using Promises first to understand the concept.',
                    'Thanks for the help everyone!'
                ];
                
                for (const content of posts) {
                    await dbRun(
                        `INSERT INTO posts (topic_id, user_id, content) VALUES (?, ?, ?)`,
                        [result.lastID, user.id, content]
                    );
                }
                console.log('Test topic and posts added');
            }
        }
        
    } catch (error) {
        console.error('Error adding test data:', error);
    }
}

module.exports = db;