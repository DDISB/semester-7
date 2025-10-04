require('module-alias/register');
import http from 'http'
import { URL } from 'url'
import { register } from '$/auth/register'
import { login } from '$/auth/login'
import { UserInput, LoginInput } from './types/index'
import { getAll } from '$/api/category';

const server = http.createServer(async (req, res) => {
  const { method, url } = req
  const parsedUrl = new URL(url || '', `http://${req.headers.host}`)
  const pathname = parsedUrl.pathname

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

  if (method === 'OPTIONS') {
    res.writeHead(200)
    res.end()
    return
  }

  // API routes
  if (method === 'POST' && pathname === '/api/register') {
    await handleRegister(req, res)
  } else if (method === 'POST' && pathname === '/api/login') {
    await handleLogin(req, res)
  } else if (method === 'GET' && pathname === '/api/categories') {
    await handleCategories(req, res)
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Route not found' }))
  }
})

async function handleRegister(req: http.IncomingMessage, res: http.ServerResponse) {
  try {
    const body = await getRequestBody(req)
    const userData: UserInput = JSON.parse(body)

    // Валидация
    if (!userData.email || !userData.username || !userData.password) {
      res.writeHead(400, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Все поля обязательны для заполнения' }))
      return
    }

    const result = await register(userData)

    res.writeHead(result.success ? 201 : 400, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(result))

  } catch (error) {
    console.error('Register handler error:', error)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
}

async function handleLogin(req: http.IncomingMessage, res: http.ServerResponse) {
  try {
    const body = await getRequestBody(req)
    const loginData: LoginInput = JSON.parse(body)

    // Валидация
    if (!loginData.email || !loginData.password) {
      res.writeHead(400, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Email и пароль обязательны' }))
      return
    }

    const result = await login(loginData)

    res.writeHead(result.success ? 200 : 401, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(result))

  } catch (error) {
    console.error('Login handler error:', error)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
}

async function handleCategories(req: http.IncomingMessage, res: http.ServerResponse) {
  try {
    const result = await getAll();

    if (result.success) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
    }
    res.end(JSON.stringify(result));
  } catch (error) {
    console.error('Categories handler error:', error);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Internal server error' }));
  }
}

function getRequestBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = ''
    req.on('data', chunk => {
      body += chunk.toString()
    })
    req.on('end', () => {
      resolve(body)
    })
    req.on('error', reject)
  })
}

const PORT = process.env.PORT || 3000

server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`)
})