export interface UserInput {
  email: string
  username: string
  password: string
}

export interface LoginInput {
  email: string
  password: string
}

export interface AuthResponse {
  success: boolean
  message: string
  user?: {
    id: string
    email: string
    username: string
    role: string
  }
  token?: string
}