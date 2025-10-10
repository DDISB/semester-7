import http from 'http'
import { URL } from 'url'
import { getSession } from '$/utils/auth'
import { IncomingMessage, ServerResponse } from 'http'
import { getAll } from './getAll'
import { createSubcategory }  from './create'
import { getRequestBody } from '$/utils/getRequestBody'
import { SubcategoryInput } from '$/types/index'


export async function handleSubcategory(
  req: IncomingMessage,
  res: ServerResponse,
  method: string | undefined,
  pathname: string) {
  if (method === 'GET' && pathname.startsWith('/api/subcategory/getAll/')) {
    const categoryId = pathname.split('/')[4]
    await handleGetALL(req, res, categoryId)
  } else if (method === 'POST' && pathname === '/api/subcategory/create') {
    await handleCreateSubcategory(req, res)
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Route not found' }))
  }
}

async function handleGetALL(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  categoryId: string) {
  try {
    const result = await getAll(categoryId)

    if (result.success) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' })
    }
    res.end(JSON.stringify(result))
  } catch (error) {
    console.error('Categories handler error:', error)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
}

async function handleCreateSubcategory(req: http.IncomingMessage, res: http.ServerResponse) {
  try {
    const body = await getRequestBody(req)
    const subcategoryData: SubcategoryInput = JSON.parse(body)
    const user = await getSession(req)
    console.log(user)

    // Валидация
    if (!subcategoryData.name || !subcategoryData.categoryId) {
      res.writeHead(400, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Имя и id картегории обязательны' }))
      return
    }

    const result = await createSubcategory(subcategoryData, user)

    if (result.success) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' })
    }
    res.end(JSON.stringify(result))
  } catch (error) {
    console.error('Categories handler error:', error)
    res.writeHead(500, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: 'Internal server error' }))
  }
}