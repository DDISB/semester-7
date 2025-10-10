import { prisma } from '$/utils/database' 
import { SubcategoriesResponse } from '$/types/index'

export async function getAll(categoryId: string): Promise<SubcategoriesResponse> {
  try {
    console.log('categoryId - ', categoryId)
    const subcategories = await prisma.subcategory.findMany({
      where: {
        categoryId
      },
      orderBy: {
        name: 'asc',
      },
    })

    if (subcategories.length === 0) {
      return {
        success: false,
        message: 'Подкатегории не найдены',
      }
    }

    return {
      success: true,
      message: 'Подкатегории успешно получены',
      subcategories: {
        data: subcategories,
        count: subcategories.length,
      },
    }
  } catch (error) {
    console.error('Get categories error:', error)
    return {
      success: false,
      message: 'Ошибка при получении подкатегории',
    }
  }
}