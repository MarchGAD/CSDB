import configparser as cp

t = cp.ConfigParser(interpolation=cp.ExtendedInterpolation())

t.read('./tmp_config.cfg')
print(t['ABC']['a'])