import { prisma } from '../utils/database'
import { UserInput, AuthResponse } from '../types/index'
import { hashPassword, generateToken } from './utils'

export async function register(userData: UserInput): Promise<AuthResponse> {
  try {
    const { email, username, password } = userData

    // Проверка существования пользователя
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [
          { email },
          { username }
        ]
      }
    })

    if (existingUser) {
      return {
        success: false,
        message: 'Пользователь с таким email или username уже существует'
      }
    }

    // Хеширование пароля
    const hashedPassword = await hashPassword(password)

    // Создание пользователя
    const user = await prisma.user.create({
      data: {
        email,
        username,
        password: hashedPassword
      }
    })

    // Генерация токена
    const token = generateToken()

    // Создание сессии
    await prisma.session.create({
      data: {
        userId: user.id,
        token,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 дней
      }
    })

    return {
      success: true,
      message: 'Пользователь успешно зарегистрирован',
      user: {
        id: user.id,
        email: user.email,
        username: user.username,
        role: user.role
      },
      token
    }

  } catch (error) {
    // console.error('Registration error:', error)
    return {
      success: false,
      message: 'Ошибка при регистрации'
    }
  }
}