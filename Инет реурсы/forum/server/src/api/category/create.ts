import { prisma } from '$/utils/database'
import { CategoryInput, CategoryResponse, AuthUser } from '$/types/index'
import { requireAdmin } from '$/utils/middleware'

export async function createCategory(
  categoryData: CategoryInput, 
  user?: AuthUser
): Promise<CategoryResponse> {
  try {
    // Проверка прав через middleware
    const adminUser = requireAdmin(user)

    const { name, description, order = 0 } = categoryData

    // Валидация order
    if (order < 0 || order > 100) {
      return {
        success: false,
        message: 'order должен быть между 0 и 100',
      }
    }

    // Проверка существующей категории
    const existingCategory = await prisma.category.findFirst({
      where: { name }
    })

    if (existingCategory) {
      return {
        success: false,
        message: 'Категория с таким названием уже существует'
      }
    }

    // Создание категории
    const category = await prisma.category.create({
      data: {
        name,
        description,
        order
      }
    })

    // Логируем действие
    console.log(`Admin ${adminUser.email} created category: ${name}`)

    return {
      success: true,
      message: 'Категория успешно создана',
      category: category
    }
  } catch (error) {
    console.error('Create category error:', error)
    
    // Обработка ошибок middleware
    if (error instanceof Error) {
      return {
        success: false,
        message: error.message,
      }
    }
    
    return {
      success: false,
      message: 'Ошибка при создании категории',
    }
  }
}