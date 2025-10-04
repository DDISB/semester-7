import { Category } from '@prisma/client';
import { prisma } from '$/utils/database';
import { CategoryInput, CategoryResponse } from '$/types/index';

export async function getAll(categoryData: CategoryInput): Promise<CategoryResponse> {
  try {
    const { name, description, order } = categoryData

    if (order < 0 || order > 100) {
      return {
        success: false,
        message: 'order должен быть между 0 и 100',
      };
    }

    const existinCategory = await prisma.category.findFirst({
      where: {
        name
      }
    })

    if (existinCategory) {
      return {
        success: false,
        message: 'Категория с таким названием уже существует'
      }
    }

    const category = await prisma.category.create({
      data: {
        name,
        description,
        order
      }
    });

    return {
      success: true,
      message: 'Категория успешно создана',
      category: {
        id: category.id,
        name: category.name,
        description: category.description,
        order: category.order,
        createdAt: category.createdAt,
        updatedAt: category.updatedAt
      },
    }
  } catch (error) {
    console.error('Get categories error:', error);
    return {
      success: false,
      message: 'Ошибка при получении категорий',
    };
  }
}