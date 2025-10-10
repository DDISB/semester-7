import { prisma } from '$/utils/database'
import { SubcategoryInput, SubcategoryResponse, AuthUser } from '$/types/index'
import { requireAuth } from '$/utils/middleware'

export async function createSubcategory(
  subcategoryData: SubcategoryInput, 
  user?: AuthUser
): Promise<SubcategoryResponse> {
  try {
    // Проверка прав через middleware
    const User = requireAuth(user)

    const { name, categoryId, description, order = 0 } = subcategoryData

    // Валидация order
    if (order < 0 || order > 100) {
      return {
        success: false,
        message: 'order должен быть между 0 и 100',
      }
    }

    // Проверка существующей категории
    const existingCategory = await prisma.category.findUnique({
      where: { id: categoryId }
    })

    if (!existingCategory) {
      return {
        success: false,
        message: 'Нет категории с указанным id'
      }
    }

    // Создание категории
    const subcategory = await prisma.subcategory.create({
      data: {
        name,
        categoryId,
        description,
        order
      }
    })

    // Логируем действие
    console.log(`User ${User.email} created category: ${name}`)

    return {
      success: true,
      message: 'Подкатегория успешно создана',
      subcategory: subcategory
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
      message: 'Ошибка при создании подкатегории',
    }
  }
}