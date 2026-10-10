import { db } from 'hatchable';
export const access='public';
export const methods=['GET','POST','PUT'];
export default async function(req,res){
 if(req.method==='GET'){const {rows}=await db.query('SELECT id,patient_id,name,age,sex,bmi,hba1c,diabetes_status,baseline_glucose,notes,created_at FROM patients ORDER BY created_at DESC');return res.json(rows)}
 const b=req.body||{};
 if(req.method==='POST'){const {rows}=await db.query('INSERT INTO patients (patient_id,name,age,sex,bmi,hba1c,diabetes_status,baseline_glucose,notes) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *',[b.patient_id,b.name,b.age||null,b.sex||null,b.bmi||null,b.hba1c||null,b.diabetes_status||'Unknown',b.baseline_glucose||null,b.notes||'']);return res.status(201).json(rows[0])}
 const {rows}=await db.query('UPDATE patients SET name=$2,age=$3,sex=$4,bmi=$5,hba1c=$6,diabetes_status=$7,baseline_glucose=$8,notes=$9 WHERE patient_id=$1 RETURNING *',[b.patient_id,b.name,b.age||null,b.sex||null,b.bmi||null,b.hba1c||null,b.diabetes_status||'Unknown',b.baseline_glucose||null,b.notes||'']);return res.json(rows[0]||null)
}