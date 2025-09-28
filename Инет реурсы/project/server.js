const path = require('path');
const db = require('./database');
const http = require('http');
const url = require('url');
const fs = require('fs');
const Auth = require('./auth');
const ForumAPI = require('./forum-api');

// Session data
const sessions = new Map();

class Server {
  constructor() {
    this.server = http.createServer(this.handleRequest.bind(this));
    this.port = 3000;
  }

  async handleRequest(req, res) {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;
    const method = req.method;

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    

    // Route handling
    if (pathname === '/' && method === 'GET') {
      await this.serveHomePage(req, res);
    } else if (pathname === '/login' && method === 'GET') {
        await this.serveLoginPage(req, res);
    } else if (pathname === '/api/login' && method === 'POST') {
        await this.handleLogin(req, res);
    } else if (pathname === '/logout' && method === 'GET') {
      await this.handleLogout(req, res);
    } else if (pathname.startsWith('/category/')) {
      await this.serveCategoryPage(req, res, parsedUrl);
    } else if (pathname.startsWith('/subcategory/')) {
      await this.serveSubcategoryPage(req, res, parsedUrl);
    } else if (pathname.startsWith('/topic/')) {
      await this.serveTopicPage(req, res, parsedUrl);
    } else if (pathname === '/api/categories' && method === 'GET') {
      await this.serveCategoriesAPI(req, res);
    } else if (pathname === '/api/posts' && method === 'POST') {
      await this.handleCreatePost(req, res);
    } else {
      this.handleNotFound(res);
    }
  }

  // Handle login form submission
  async handleLogin(req, res) {
    try {
        const body = await this.getRequestBody(req);
        const { login, password, returnUrl = '/' } = JSON.parse(body);

        if (!login || !password) {
            this.sendResponse(res, 400, { 
                success: false, 
                message: 'Login and password are required' 
            });
            return;
        }

        const result = await Auth.authenticate(login, password);
        
        if (result.success) {
            // Create session
            const sessionId = this.generateSessionId();
            sessions.set(sessionId, {
                userId: result.user.id,
                login: result.user.login,
                createdAt: Date.now()
            });
            
            // Set session cookie
            res.setHeader('Set-Cookie', `sessionId=${sessionId}; HttpOnly; Path=/; Max-Age=3600`);
            
            // Return success with the returnUrl
            this.sendResponse(res, 200, {
                ...result,
                returnUrl: returnUrl
            });
        } else {
            this.sendResponse(res, 401, result);
        }

    } catch (error) {
        console.error('Login processing error:', error);
        this.sendResponse(res, 500, { 
            success: false, 
            message: 'Internal server error' 
        });
    }
}

  // Handle logout
  async handleLogout(req, res) {
    const cookies = this.parseCookies(req);
    const sessionId = cookies.sessionId;

    if (sessionId) {
      sessions.delete(sessionId);
    }

    // Clear session cookie
    res.setHeader('Set-Cookie', 'sessionId=; HttpOnly; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT');

    // Redirect to login page 
    res.writeHead(302, { 'Location': '/login' });
    res.end();
  }

  // Helper methods
  getRequestBody(req) {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => body += chunk.toString());
      req.on('end', () => resolve(body));
      req.on('error', reject);
    });
  }

  sendResponse(res, statusCode, data) {
    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  }

  generateSessionId() {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  parseCookies(req) {
    const cookies = {};
    const cookieHeader = req.headers.cookie;
    if (cookieHeader) {
      cookieHeader.split(';').forEach(cookie => {
        const parts = cookie.split('=');
        cookies[parts[0].trim()] = parts[1]?.trim();
      });
    }
    return cookies;
  }

  handleNotFound(res) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found');
  }

  start() {
    this.server.listen(this.port, () => {
      console.log(`Server running at http://localhost:${this.port}`);
      console.log('Test user: testuser / password123');
    });
  }

  async serveHomePage(req, res) {
    try {
      const categories = await ForumAPI.getCategories();
      const cookies = this.parseCookies(req);
      const sessionId = cookies.sessionId;
      let isLoggedIn = false;
      if (sessionId && sessions.has(sessionId)) {
        isLoggedIn = true;
      }
      const html = await this.generateHomePage(categories, isLoggedIn);
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } catch (error) {
      console.error('Error serving home page:', error);
      this.sendResponse(res, 500, { error: 'Internal server error' });
    }
  }

  async serveLoginPage(req, res) {
    const parsedUrl = url.parse(req.url, true);
    const returnUrl = parsedUrl.query.returnUrl || '/';
    
    const html = this.generateLoginPage(returnUrl);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
}

  async serveCategoryPage(req, res, parsedUrl) {
    try {
      const categoryId = parseInt(parsedUrl.pathname.split('/')[2]);
      const subcategories = await ForumAPI.getSubcategories(categoryId);
      const html = await this.generateCategoryPage(categoryId, subcategories);
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } catch (error) {
      console.error('Error serving category page:', error);
      this.handleNotFound(res);
    }
  }

  async serveSubcategoryPage(req, res, parsedUrl) {
    try {
      const subcategoryId = parseInt(parsedUrl.pathname.split('/')[2]);
      const page = parseInt(parsedUrl.query.page) || 1;

      const [subcategory, topics] = await Promise.all([
        ForumAPI.getSubcategory(subcategoryId),
        ForumAPI.getTopics(subcategoryId, page)
      ]);

      const html = await this.generateSubcategoryPage(subcategory, topics, page);
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } catch (error) {
      console.error('Error serving subcategory page:', error);
      this.handleNotFound(res);
    }
  }

  async serveTopicPage(req, res, parsedUrl) {
    try {
      const topicId = parseInt(parsedUrl.pathname.split('/')[2]);
      const page = parseInt(parsedUrl.query.page) || 1;

      const [topic, posts] = await Promise.all([
        ForumAPI.getTopic(topicId),
        ForumAPI.getPosts(topicId, page)
      ]);

      const cookies = this.parseCookies(req);
      const sessionId = cookies.sessionId;
      let isLoggedIn = false;
      if (sessionId && sessions.has(sessionId)) {
        isLoggedIn = true;
      }

      // Increment view count
      await ForumAPI.incrementViewCount(topicId);

      const html = await this.generateTopicPage(topic, posts, page, isLoggedIn);
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(html);
    } catch (error) {
      console.error('Error serving topic page:', error);
      this.handleNotFound(res);
    }
  }

  async generateHomePage(categories, isLoggedIn = false) {
    const authSection = isLoggedIn
        ? `<div style="text-align: right; margin-bottom: 20px;"> 
             <a href="/logout" style="margin-left: 10px;">Выйти из аккаунта</a>
           </div>`
        : `<div style="text-align: right; margin-bottom: 20px;">
             <a href="/login">Войти</a>
           </div>`;

    const categoriesHtml = categories.map(cat => `
          <div class="category">
              <h3><a href="/category/${cat.id}">${cat.name}</a></h3>
              <p>${cat.description}</p>
          </div>
      `).join('');

    return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Forum Home</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .category { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .subcategory { margin: 15px 0; padding: 15px; border: 1px solid #eee; margin-left: 20px; }
                .topic { margin: 10px 0; padding: 10px; background: #f9f9f9; margin-left: 40px; }
                .post { margin: 10px 0; padding: 10px; border-bottom: 1px solid #eee; }
                .pagination { margin: 20px 0; }
                .pagination a { margin: 0 5px; padding: 5px 10px; border: 1px solid #ddd; text-decoration: none; }
                .pagination .current { background: #007bff; color: white; }
                .breadcrumb { margin: 10px 0; color: #666; }
                .breadcrumb a { color: #007bff; text-decoration: none; }
                .header { display: flex; aligne-items: center;}
            </style>
        </head>
        <body>
            <div class="header"><h1>Forum Categories</h1> ${authSection}</div>
            ${categoriesHtml}
        </body>
        </html>`;
  }

  generateLoginPage(returnUrl = '/') {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Forum</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 400px;
            margin: 100px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .login-form {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: bold;
        }
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            display: none;
        }
        .success {
            color: #155724;
            background-color: #d4edda;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            display: none;
        }
        .back-link {
            text-align: center;
            margin-top: 20px;
        }
        .return-info {
            background: #e7f3ff;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #0066cc;
        }
    </style>
</head>
<body>
    <div class="login-form">
        <h2>Login to Forum</h2>
        
        ${returnUrl !== '/' ? `
        <div class="return-info">
            After login you will be returned to the previous page
        </div>
        ` : ''}
        
        <div id="errorMessage" class="error"></div>
        <div id="successMessage" class="success"></div>
        
        <form id="loginForm">
            <input type="hidden" id="returnUrl" value="${returnUrl}">
            <div class="form-group">
                <label for="login">Username:</label>
                <input type="text" id="login" name="login" required>
            </div>
            
            <div class="form-group">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            
            <button type="submit">Login</button>
        </form>
        
        <div class="back-link">
            <a href="${returnUrl}">← Back to previous page</a>
        </div>
    </div>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const login = document.getElementById('login').value;
            const password = document.getElementById('password').value;
            const returnUrl = document.getElementById('returnUrl').value;
            const errorDiv = document.getElementById('errorMessage');
            const successDiv = document.getElementById('successMessage');
            
            // Hide previous messages
            errorDiv.style.display = 'none';
            successDiv.style.display = 'none';
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        login, 
                        password,
                        returnUrl 
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    successDiv.textContent = 'Login successful! Redirecting...';
                    successDiv.style.display = 'block';
                    
                    // Store user data in localStorage
                    localStorage.setItem('user', JSON.stringify(result.user));
                    
                    // Redirect to returnUrl after 1 second
                    setTimeout(() => {
                        window.location.href = returnUrl;
                    }, 1000);
                } else {
                    errorDiv.textContent = result.message;
                    errorDiv.style.display = 'block';
                }
            } catch (error) {
                errorDiv.textContent = 'Network error. Please try again.';
                errorDiv.style.display = 'block';
            }
        });
        
        // Focus on login field when page loads
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('login').focus();
        });
    </script>
</body>
</html>`;
}

  async generateCategoryPage(categoryId, subcategories) {
    if (subcategories.length === 0) {
      return this.generateErrorPage('Category not found');
    }

    const subcategoriesHtml = subcategories.map(sub => `
        <div class="subcategory">
            <h4><a href="/subcategory/${sub.id}">${sub.name}</a></h4>
            <p>${sub.description || 'No description'}</p>
        </div>
    `).join('');

    return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>${subcategories[0].category_name} - Forum</title>
        <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .category { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .subcategory { margin: 15px 0; padding: 15px; border: 1px solid #eee; margin-left: 20px; }
                .topic { margin: 10px 0; padding: 10px; background: #f9f9f9; margin-left: 40px; }
                .post { margin: 10px 0; padding: 10px; border-bottom: 1px solid #eee; }
                .pagination { margin: 20px 0; }
                .pagination a { margin: 0 5px; padding: 5px 10px; border: 1px solid #ddd; text-decoration: none; }
                .pagination .current { background: #007bff; color: white; }
                .breadcrumb { margin: 10px 0; color: #666; }
                .breadcrumb a { color: #007bff; text-decoration: none; }
            </style>
    </head>
    <body>
        <div class="breadcrumb">
            <a href="/">Home</a> > <span>${subcategories[0].category_name}</span>
        </div>
        <h1>${subcategories[0].category_name}</h1>
        ${subcategoriesHtml}
        <p><a href="/">← Back to Home</a></p>
    </body>
    </html>`;
  }

  async generateSubcategoryPage(subcategory, topics, currentPage = 1) {
    if (!subcategory) {
      return this.generateErrorPage('Subcategory not found');
    }

    const topicsHtml = topics.map(topic => `
        <div class="topic">
            <h4><a href="/topic/${topic.id}">${topic.title}</a></h4>
            <p>By ${topic.author} • ${new Date(topic.created_at).toLocaleDateString()} • 
               Replies: ${(topic.post_count || 1) - 1} • Views: ${topic.view_count || 0}</p>
        </div>
    `).join('');

    const paginationHtml = this.generatePagination(currentPage, Math.ceil(topics.length / 20), `/subcategory/${subcategory.id}`);

    return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>${subcategory.name} - Forum</title>
        <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .category { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .subcategory { margin: 15px 0; padding: 15px; border: 1px solid #eee; margin-left: 20px; }
                .topic { margin: 10px 0; padding: 10px; background: #f9f9f9; margin-left: 40px; }
                .post { margin: 10px 0; padding: 10px; border-bottom: 1px solid #eee; }
                .pagination { margin: 20px 0; }
                .pagination a { margin: 0 5px; padding: 5px 10px; border: 1px solid #ddd; text-decoration: none; }
                .pagination .current { background: #007bff; color: white; }
                .breadcrumb { margin: 10px 0; color: #666; }
                .breadcrumb a { color: #007bff; text-decoration: none; }
            </style>
    </head>
    <body>
        <div class="breadcrumb">
            <a href="/">Home</a> > 
            <a href="/category/${subcategory.category_id}">${subcategory.category_name}</a> > 
            <span>${subcategory.name}</span>
        </div>
        <h1>${subcategory.name}</h1>
        <p>${subcategory.description || ''}</p>
        
        <h2>Topics</h2>
        ${topicsHtml || '<p>No topics yet.</p>'}
        
        ${paginationHtml}
        
        <p><a href="/category/${subcategory.category_id}">← Back to ${subcategory.category_name}</a></p>
    </body>
    </html>`;
  }

  async generateTopicPage(topic, posts, currentPage = 1, isLoggedIn = false) {
    if (!topic) {
        return this.generateErrorPage('Topic not found');
    }

    const postsHtml = posts.map((post, index) => `
        <div class="post" id="post-${post.id}">
            <div class="post-header">
                <strong>${post.author}</strong> • 
                #${index + 1 + ((currentPage - 1) * 20)} • 
                ${new Date(post.created_at).toLocaleString()}
                ${post.is_edited ? '(edited)' : ''}
            </div>
            <div class="post-content">
                ${post.content.replace(/\n/g, '<br>')}
            </div>
        </div>
    `).join('');

    // Получаем общее количество постов для пагинации
    const totalPosts = await this.getTotalPostsCount(topic.id);
    const totalPages = Math.ceil(totalPosts / 20);
    const paginationHtml = this.generatePagination(currentPage, totalPages, `/topic/${topic.id}`);

    // Форма ответа в зависимости от авторизации
    const replySection = isLoggedIn ? `
        <div style="margin-top: 30px; padding: 15px; background: #f5f5f5;">
            <h3>Post a Reply</h3>
            <form id="replyForm" onsubmit="return submitReply(${topic.id})">
                <textarea id="replyContent" rows="4" style="width: 100%; margin: 10px 0;" placeholder="Write your reply..." required></textarea><br>
                <button type="submit">Post Reply</button>
            </form>
        </div>
    ` : `
        <div style="margin-top: 30px; padding: 15px; background: #f5f5f5; text-align: center;">
            <p><a href="/login">Login to your account</a> to post a reply</p>
        </div>
    `;

    // Секция авторизации
    const authSection = isLoggedIn ? `
        <div style="text-align: right; margin-bottom: 20px;">
            <a href="/logout" style="margin-left: 10px;">Logout</a>
        </div>
    ` : `
        <div style="text-align: right; margin-bottom: 20px;">
            <a href="/login">Login</a> to participate in discussions
        </div>
    `;

    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>${topic.title} - Forum</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .category { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
            .subcategory { margin: 15px 0; padding: 15px; border: 1px solid #eee; margin-left: 20px; }
            .topic { margin: 10px 0; padding: 10px; background: #f9f9f9; margin-left: 40px; }
            .post { margin: 10px 0; padding: 10px; border-bottom: 1px solid #eee; }
            .pagination { margin: 20px 0; text-align: center; }
            .pagination a, .pagination span { 
                margin: 0 5px; 
                padding: 5px 10px; 
                border: 1px solid #ddd; 
                text-decoration: none;
                display: inline-block;
            }
            .pagination a:hover { background: #f0f0f0; }
            .pagination .current { background: #007bff; color: white; }
            .breadcrumb { margin: 10px 0; color: #666; }
            .breadcrumb a { color: #007bff; text-decoration: none; }
            .post-header { color: #666; font-size: 0.9em; margin-bottom: 5px; }
            .post-content { margin: 10px 0; }
        </style>
    </head>
    <body>
        ${authSection}
        
        <div class="breadcrumb">
            <a href="/">Home</a> > 
            <a href="/category/${topic.category_id}">${topic.category_name}</a> > 
            <a href="/subcategory/${topic.subcategory_id}">${topic.subcategory_name}</a> > 
            <span>${topic.title}</span>
        </div>
        
        <h1>${topic.title}</h1>
        <p>By ${topic.author} • ${new Date(topic.created_at).toLocaleDateString()} • Views: ${topic.view_count || 0}</p>
        
        <h2>Posts (${totalPosts} total)</h2>
        ${postsHtml}
        
        ${paginationHtml}
        
        ${topic.is_locked ? `
            <div style="margin-top: 30px; padding: 15px; background: #ffe6e6; text-align: center;">
                <p>This topic is locked. No new replies can be posted.</p>
            </div>
        ` : replySection}
        
        <p><a href="/subcategory/${topic.subcategory_id}">← Back to ${topic.subcategory_name}</a></p>
        
        <script>
            function submitReply(topicId) {
                const content = document.getElementById('replyContent').value.trim();
                if (!content) {
                    alert('Please enter your reply');
                    return false;
                }
                
                // Отправка сообщения на сервер
                fetch('/api/posts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        topicId: topicId,
                        content: content
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Обновляем страницу после успешной отправки
                        window.location.reload();
                    } else {
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error posting reply');
                });
                
                return false;
            }
            
            // Функция для перехода по страницам
            function goToPage(page) {
                window.location.href = '/topic/${topic.id}?page=' + page;
            }
            
            // Показываем информацию о текущей странице
            document.addEventListener('DOMContentLoaded', function() {
                const pageInfo = document.createElement('div');
                pageInfo.style.textAlign = 'center';
                pageInfo.style.margin = '10px 0';
                pageInfo.style.color = '#666';
                pageInfo.innerHTML = 'Page ${currentPage} of ${totalPages}';
                document.querySelector('.pagination').parentNode.insertBefore(pageInfo, document.querySelector('.pagination'));
            });
        </script>
    </body>
    </html>`;
  }

  // Вспомогательный метод для получения общего количества постов
  async getTotalPostsCount(topicId) {
      return new Promise((resolve, reject) => {
          db.get('SELECT COUNT(*) as count FROM posts WHERE topic_id = ?', [topicId], (err, result) => {
              if (err) reject(err);
              else resolve(result.count);
          });
      });
  }

  // Обновленный метод пагинации
  generatePagination(currentPage, totalPages, baseUrl) {
      if (totalPages <= 1) return '';
      
      let paginationHtml = '<div class="pagination">';
      
      // Previous page
      if (currentPage > 1) {
          paginationHtml += `<a href="${baseUrl}?page=${currentPage - 1}">← Previous</a>`;
      }
      
      // Page numbers
      const maxVisiblePages = 5;
      let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
      let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
      
      // Adjust if we're at the end
      if (endPage - startPage + 1 < maxVisiblePages) {
          startPage = Math.max(1, endPage - maxVisiblePages + 1);
      }
      
      // First page and ellipsis
      if (startPage > 1) {
          paginationHtml += `<a href="${baseUrl}?page=1">1</a>`;
          if (startPage > 2) {
              paginationHtml += '<span>...</span>';
          }
      }
      
      // Page numbers
      for (let i = startPage; i <= endPage; i++) {
          if (i === currentPage) {
              paginationHtml += `<span class="current">${i}</span>`;
          } else {
              paginationHtml += `<a href="${baseUrl}?page=${i}">${i}</a>`;
          }
      }
      
      // Last page and ellipsis
      if (endPage < totalPages) {
          if (endPage < totalPages - 1) {
              paginationHtml += '<span>...</span>';
          }
          paginationHtml += `<a href="${baseUrl}?page=${totalPages}">${totalPages}</a>`;
      }
      
      // Next page
      if (currentPage < totalPages) {
          paginationHtml += `<a href="${baseUrl}?page=${currentPage + 1}">Next →</a>`;
      }
      
      paginationHtml += '</div>';
      return paginationHtml;
  }

  generateErrorPage(message) {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error - Forum</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        </style>
    </head>
    <body>
        <h1>Error</h1>
        <p>${message}</p>
        <p><a href="/">Go to Homepage</a></p>
    </body>
    </html>`;
  }

  // Новый метод для обработки создания постов
async handleCreatePost(req, res) {
    try {
        const cookies = this.parseCookies(req);
        const sessionId = cookies.sessionId;
        
        if (!sessionId || !sessions.has(sessionId)) {
            this.sendResponse(res, 401, { success: false, message: 'Not authorized' });
            return;
        }
        
        const sessionData = sessions.get(sessionId);
        const body = await this.getRequestBody(req);
        const { topicId, content } = JSON.parse(body);
        
        if (!topicId || !content) {
            this.sendResponse(res, 400, { success: false, message: 'Topic ID and content are required' });
            return;
        }
        
        // Проверяем, существует ли тема и не заблокирована ли она
        const topic = await ForumAPI.getTopic(topicId);
        if (!topic) {
            this.sendResponse(res, 404, { success: false, message: 'Topic not found' });
            return;
        }
        
        if (topic.is_locked) {
            this.sendResponse(res, 403, { success: false, message: 'Topic is locked' });
            return;
        }
        
        // Создаем пост
        await ForumAPI.addPost(topicId, sessionData.userId, content);
        
        this.sendResponse(res, 200, { 
            success: true, 
            message: 'Post created successfully' 
        });
        
    } catch (error) {
        console.error('Error creating post:', error);
        this.sendResponse(res, 500, { success: false, message: 'Internal server error' });
    }
}

  async serveCategoriesAPI(req, res) {
    try {
      const categories = await ForumAPI.getCategories();
      this.sendResponse(res, 200, categories);
    } catch (error) {
      console.error('Error serving categories API:', error);
      this.sendResponse(res, 500, { error: 'Internal server error' });
    }
  }
}

const server = new Server();
server.start();