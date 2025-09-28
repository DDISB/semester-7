const db = require('./database');

class ForumAPI {
    // Get all categories (first level)
    static getCategories() {
        return new Promise((resolve, reject) => {
            db.all(`SELECT * FROM categories ORDER BY position`, (err, categories) => {
                if (err) reject(err);
                else resolve(categories);
            });
        });
    }

    // Get subcategories for a category
    static getSubcategories(categoryId) {
        return new Promise((resolve, reject) => {
            db.all(`SELECT s.*, c.name as category_name 
                   FROM subcategories s 
                   JOIN categories c ON s.category_id = c.id 
                   WHERE s.category_id = ? 
                   ORDER BY s.position`, 
                [categoryId], (err, subcategories) => {
                    if (err) reject(err);
                    else resolve(subcategories);
                });
        });
    }

    // Get topics for a subcategory with pagination
    static getTopics(subcategoryId, page = 1, limit = 20) {
        return new Promise((resolve, reject) => {
            const offset = (page - 1) * limit;
            
            db.all(`SELECT t.*, u.login as author, 
                   (SELECT COUNT(*) FROM posts p WHERE p.topic_id = t.id) as post_count,
                   (SELECT MAX(created_at) FROM posts p WHERE p.topic_id = t.id) as last_post_date
                   FROM topics t 
                   JOIN users u ON t.user_id = u.id 
                   WHERE t.subcategory_id = ? 
                   ORDER BY t.is_pinned DESC, last_post_date DESC 
                   LIMIT ? OFFSET ?`,
                [subcategoryId, limit, offset], (err, topics) => {
                    if (err) reject(err);
                    else resolve(topics);
                });
        });
    }

    // Get posts for a topic with pagination
    static getPosts(topicId, page = 1, limit = 20) {
        return new Promise((resolve, reject) => {
            const offset = (page - 1) * limit;
            
            db.all(`SELECT p.*, u.login as author 
                   FROM posts p 
                   JOIN users u ON p.user_id = u.id 
                   WHERE p.topic_id = ? 
                   ORDER BY p.created_at ASC 
                   LIMIT ? OFFSET ?`,
                [topicId, limit, offset], (err, posts) => {
                    if (err) reject(err);
                    else resolve(posts);
                });
        });
    }

    // Get topic details
    static getTopic(topicId) {
        return new Promise((resolve, reject) => {
            db.get(`SELECT t.*, u.login as author, s.name as subcategory_name,
                   c.name as category_name, s.category_id
                   FROM topics t 
                   JOIN users u ON t.user_id = u.id 
                   JOIN subcategories s ON t.subcategory_id = s.id 
                   JOIN categories c ON s.category_id = c.id 
                   WHERE t.id = ?`, 
                [topicId], (err, topic) => {
                    if (err) reject(err);
                    else resolve(topic);
                });
        });
    }

    // Get subcategory details
    static getSubcategory(subcategoryId) {
        return new Promise((resolve, reject) => {
            db.get(`SELECT s.*, c.name as category_name 
                   FROM subcategories s 
                   JOIN categories c ON s.category_id = c.id 
                   WHERE s.id = ?`, 
                [subcategoryId], (err, subcategory) => {
                    if (err) reject(err);
                    else resolve(subcategory);
                });
        });
    }

    // Add a new post
    static addPost(topicId, userId, content) {
        return new Promise((resolve, reject) => {
            db.run(`INSERT INTO posts (topic_id, user_id, content) VALUES (?, ?, ?)`,
                [topicId, userId, content], function(err) {
                    if (err) reject(err);
                    else resolve({ id: this.lastID });
                });
        });
    }

    // Add a new topic
    static addTopic(subcategoryId, userId, title, content) {
        return new Promise((resolve, reject) => {
            db.run(`INSERT INTO topics (subcategory_id, user_id, title, content) VALUES (?, ?, ?, ?)`,
                [subcategoryId, userId, title, content], function(err) {
                    if (err) reject(err);
                    else resolve({ id: this.lastID });
                });
        });
    }

    // Increment topic view count
    static incrementViewCount(topicId) {
        return new Promise((resolve, reject) => {
            db.run(`UPDATE topics SET view_count = view_count + 1 WHERE id = ?`,
                [topicId], (err) => {
                    if (err) reject(err);
                    else resolve();
                });
        });
    }
}

module.exports = ForumAPI;