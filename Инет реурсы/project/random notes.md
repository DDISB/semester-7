# для проверки логина в powershell
iwr -Uri "http://localhost:3000/login" -Method POST -ContentType "application/json" -Body '{"login":"testuser","password":"password123"}'

