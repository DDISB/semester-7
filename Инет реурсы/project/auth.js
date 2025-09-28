const bcrypt = require('bcryptjs');
const db = require('./database');

class Auth {
    // Check login and password
    static async authenticate(login, password) {
        return new Promise((resolve, reject) => {
            db.get('SELECT * FROM users WHERE login = ?', [login], (err, user) => {
                if (err) {
                    reject(err);
                } else if (!user) {
                    resolve({ success: false, message: 'User not found' });
                } else {
                    // Compare password hash
                    const isValid = bcrypt.compareSync(password, user.password);
                    if (isValid) {
                        resolve({ 
                            success: true, 
                            message: 'Authorization successful',
                            user: { id: user.id, login: user.login }
                        });
                    } else {
                        resolve({ success: false, message: 'Invalid password' });
                    }
                }
            });
        });
    }

    // Create new user (for future use)
    static async createUser(login, password) {
        return new Promise((resolve, reject) => {
            const hashedPassword = bcrypt.hashSync(password, 10);
            
            db.run('INSERT INTO users (login, password) VALUES (?, ?)', 
                [login, hashedPassword], 
                function(err) {
                    if (err) {
                        reject(err);
                    } else {
                        resolve({ 
                            success: true, 
                            message: 'User created successfully',
                            userId: this.lastID 
                        });
                    }
                }
            );
        });
    }
}

module.exports = Auth;