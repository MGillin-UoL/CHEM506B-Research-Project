driver = IKADriver(serial_port='/dev/ttyACM0')
 
driver.setStir(1000)
driver.startStir()
 
driver.getStirringSpeed()
print(driver.getStirringSpeed())
time.sleep(5)
 
driver.getStirringSpeed()
print(driver.getStirringSpeed())
 
time.sleep(30)
driver.stopStir()
