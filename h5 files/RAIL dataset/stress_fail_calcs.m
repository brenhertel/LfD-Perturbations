stress = [22.6354 33.5758 0;
          33.5758 0 0;
          0 0 0];
p_stress = eig(stress);

yield = 280;

max_normal = max(abs(p_stress))
max_shear = 0.5 * abs(max(p_stress) - min(p_stress))
max_von_mises = sqrt(0.5 * ((p_stress(1) - p_stress(2))^2 + (p_stress(1) - p_stress(3))^2 + (p_stress(3) - p_stress(2))^2))

normal_FS = yield / max_normal
shear_FS = (yield / 2) / max_shear
shear_von_mises = yield / max_von_mises